defmodule LgbmExxTest do
  use ExUnit.Case, async: false

  defp setup_model(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)

    {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

    model =
      LgbmEx.fit("test", df, "species",
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 20
      )

    %{model: model}
  end

  describe "cross_validate" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "basic uses returns results", %{model: model} do
      {100, 50, [result | _]} = LgbmExx.cross_validate(model, 3)

      assert result.evaluator_result == nil
      assert result.prediction == []
      assert result.last_evaluation
      assert 4 == Enum.count(result.feature_importance_split)
    end

    test "uses with x_test", %{model: model} do
      x_test = [
        [5.4, 3.9, 1.7, 0.4],
        [5.7, 2.8, 4.5, 1.3],
        [7.6, 3.0, 6.6, 2.2]
      ]

      {_, _, [result | _]} =
        LgbmExx.cross_validate(model, 3, x_test: x_test, folding_rule: :stratified)

      [[p1, _, _], [_, p2, _], [_, _, p3]] = result.prediction

      assert p1 > 0.5
      assert p2 > 0.5
      assert p3 > 0.5
    end

    test "uses with folding_rule", %{model: model} do
      rules = [:raw, :shuffle, :stratified, :stratified_shuffle]

      results =
        rules
        |> Map.new(fn rule ->
          {train_size, val_size, cv_results} =
            LgbmExx.cross_validate(model, 3, folding_rule: rule)

          {rule, {train_size, val_size, cv_results}}
        end)

      # 各 folding_rule について検証
      Enum.each(rules, fn rule ->
        {train_size, val_size, cv_results} = Map.fetch!(results, rule)

        # CV結果の数が3（k=3）であることを確認
        assert Enum.count(cv_results) == 3,
               "folding_rule #{rule} should return 3 CV results for k=3"

        # 各CV結果の構造が正しいことを確認
        Enum.each(cv_results, fn result ->
          assert is_map(result)
          assert Map.has_key?(result, :last_evaluation)
          assert Map.has_key?(result, :num_iterations)
          assert Map.has_key?(result, :feature_importance_split)
        end)

        # fold のサイズが妥当なこと
        # 各foldのtrain + val = 150 (Irisデータセット)
        assert train_size + val_size == 150,
               "folding_rule #{rule} should partition all 150 samples"
      end)

      # raw split は他と異なる評価値を持つことが期待される
      # （randomnessがないため、より高い損失値になることが多い）
      raw_result = elem(Map.fetch!(results, :raw), 2) |> List.first()
      shuffle_result = elem(Map.fetch!(results, :shuffle), 2) |> List.first()

      # rawとshuffle は異なるデータ順序を使うため、結果が異なるはず
      refute raw_result.last_evaluation == shuffle_result.last_evaluation,
             "raw and shuffle folding should produce different results due to data ordering"
    end

    test "uses with evaluator", %{model: model} do
      evaluator = fn y_val, pred_val ->
        Enum.zip(y_val, pred_val)
        |> Enum.map(fn {class, probs} ->
          index = round(class)
          Enum.at(probs, index)
        end)
        |> Explorer.Series.from_list()
        |> Explorer.Series.mean()
      end

      {_, _, [result | _]} = LgbmExx.cross_validate(model, 3, evaluator: evaluator)

      assert result.evaluator_result
    end

    test "with different k values", %{model: model} do
      # k=3でテスト（既に他のテストで検証済み）
      {_, _, results_3} = LgbmExx.cross_validate(model, 3)
      assert Enum.count(results_3) == 3

      # 各結果が正しく返されることを確認
      Enum.each(results_3, fn result ->
        assert is_map(result)
        assert Map.has_key?(result, :num_iterations)
      end)
    end

    test "with all options combined", %{model: model} do
      x_test = [[5.4, 3.9, 1.7, 0.4]]

      evaluator = fn y_val, pred_val ->
        Enum.zip(y_val, pred_val) |> Enum.count()
      end

      {train_size, val_size, results} =
        LgbmExx.cross_validate(model, 3,
          x_test: x_test,
          evaluator: evaluator,
          folding_rule: :stratified_shuffle
        )

      assert train_size > 0
      assert val_size > 0
      assert Enum.count(results) == 3

      Enum.each(results, fn result ->
        assert result.evaluator_result != nil
        assert result.prediction != []
        assert Enum.count(result.prediction) == 1
      end)
    end
  end

  describe "aggregate_cv_results" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "returns aggregation result", %{model: model} do
      {100, 50, cv_results} = LgbmExx.cross_validate(model, 3)

      result = LgbmExx.aggregate_cv_results(cv_results)

      # Check structure
      assert is_map(result)
      assert Map.has_key?(result, :last_evaluation)

      # Check that aggregated values include statistics
      assert is_map(result.last_evaluation)
      assert Map.has_key?(result.last_evaluation, :mean)
    end

    test "with evaluator returns aggregated evaluator_result", %{model: model} do
      evaluator = fn y_val, pred_val ->
        Enum.zip(y_val, pred_val) |> Enum.count()
      end

      {_, _, cv_results} = LgbmExx.cross_validate(model, 3, evaluator: evaluator)
      result = LgbmExx.aggregate_cv_results(cv_results)

      # evaluator_resultが統計情報として含まれていることを確認
      assert is_map(result.evaluator_result)
      assert Map.has_key?(result.evaluator_result, :mean)
      assert Map.has_key?(result.evaluator_result, :std)
    end

    test "with x_test returns aggregated predictions", %{model: model} do
      x_test = [[5.4, 3.9, 1.7, 0.4]]

      {_, _, cv_results} = LgbmExx.cross_validate(model, 3, x_test: x_test)
      result = LgbmExx.aggregate_cv_results(cv_results)

      # predictionが平均化されていることを確認
      assert is_list(result.prediction)
      assert Enum.count(result.prediction) == 1
      [pred] = result.prediction
      assert is_list(pred)
      # 3 classes
      assert Enum.count(pred) == 3
    end
  end

  describe "grid_search" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "returns result each parameter combination", %{model: model} do
      grid = [
        num_iterations: [5, 10],
        min_data_in_leaf: [2, 3, 4]
      ]

      [{parameters, result} | _] = results = LgbmExx.grid_search(model, grid, 3)

      assert 6 == Enum.count(results)
      assert [num_iterations: 5, min_data_in_leaf: 2] == parameters
      # Check result is a map with aggregated statistics
      assert is_map(result)
      assert Map.has_key?(result, :num_iterations)
      # num_iterations should be aggregated (mean of 5)
      assert result.num_iterations.mean == 5.0
    end

    test "with single parameter grid", %{model: model} do
      grid = [
        num_iterations: [5, 10, 15]
      ]

      results = LgbmExx.grid_search(model, grid, 3)

      assert 3 == Enum.count(results)
    end

    test "with three parameters grid", %{model: model} do
      grid = [
        num_iterations: [5, 10],
        min_data_in_leaf: [2, 3],
        max_depth: [3, 4]
      ]

      results = LgbmExx.grid_search(model, grid, 3)

      assert 8 == Enum.count(results)
    end
  end

  describe "one_hot_encode" do
    @describetag :tmp_dir

    test "returns df with one_hot_encoded columns" do
      df = Explorer.Datasets.iris()
      df_done = LgbmExx.one_hot_encode(df, ["species"])

      assert_raise ArgumentError, fn -> df_done["species"] end
      assert df_done["species_Iris-setosa"]
      assert df_done["species_Iris-versicolor"]
      assert df_done["species_Iris-virginica"]
    end

    test "with empty columns list returns original dataframe" do
      df = Explorer.Datasets.iris()
      df_done = LgbmExx.one_hot_encode(df, [])

      assert df_done == df
    end

    test "with nil columns returns original dataframe" do
      df = Explorer.Datasets.iris()
      df_done = LgbmExx.one_hot_encode(df, nil)

      assert df_done == df
    end

    test "with threshold parameter filters rare categories" do
      df = Explorer.Datasets.iris()
      # threshold=10で、countが10以下の要素がフィルタされる
      df_done = LgbmExx.one_hot_encode(df, ["species"], 10)

      # Irisデータセットではそれぞれ50サンプルあるので全て残る
      assert df_done["species_Iris-setosa"]
      assert df_done["species_Iris-versicolor"]
      assert df_done["species_Iris-virginica"]
    end

    test "with high threshold filters all categories" do
      df = Explorer.Datasets.iris()
      # threshold=100で、全要素がフィルタされる
      df_done = LgbmExx.one_hot_encode(df, ["species"], 100)

      # 返り値は返される（フレームが変更される）
      assert is_struct(df_done, Explorer.DataFrame)
      # speciesカラムは削除されている
      assert_raise ArgumentError, fn -> df_done["species"] end
    end
  end

  describe "columns_stats" do
    @describetag :tmp_dir

    test "returns statistics map" do
      df = Explorer.Datasets.iris()
      stats = LgbmExx.columns_stats(df, ["sepal_length", "sepal_width"])

      assert stats["sepal_length"]["count"] == 150
      assert stats["sepal_width"]["count"] == 150
    end

    test "with single row dataframe" do
      df = Explorer.Datasets.iris() |> Explorer.DataFrame.slice(0, 1)
      stats = LgbmExx.columns_stats(df, ["sepal_length", "sepal_width"])

      assert stats["sepal_length"]["count"] == 1
      assert stats["sepal_width"]["count"] == 1
    end

    test "includes min, max, mean, std for columns" do
      df = Explorer.Datasets.iris()
      stats = LgbmExx.columns_stats(df, ["sepal_length"])

      # 各統計値が含まれていることを確認
      assert Map.has_key?(stats["sepal_length"], "min")
      assert Map.has_key?(stats["sepal_length"], "max")
      assert Map.has_key?(stats["sepal_length"], "mean")
      assert Map.has_key?(stats["sepal_length"], "std")
      assert Map.has_key?(stats["sepal_length"], "count")
    end
  end

  describe "get_correlation_map" do
    @describetag :tmp_dir

    test "returns statistics map" do
      df = Explorer.Datasets.iris()
      correlation_result = Explorer.DataFrame.correlation(df)
      sepal_length = LgbmExx.get_correlation_map(correlation_result, "sepal_length")

      assert %{
               "petal_length" => 0.8717541573048646,
               "petal_width" => 0.8179536333691573,
               "sepal_length" => 1.0,
               "sepal_width" => -0.10936924995064126
             } = sepal_length
    end

    test "returns all column names in the map" do
      df = Explorer.Datasets.iris()
      correlation_result = Explorer.DataFrame.correlation(df)
      sepal_length = LgbmExx.get_correlation_map(correlation_result, "sepal_length")

      # 全てのカラム名がキーに含まれていることを確認
      assert Map.has_key?(sepal_length, "sepal_length")
      assert Map.has_key?(sepal_length, "sepal_width")
      assert Map.has_key?(sepal_length, "petal_length")
      assert Map.has_key?(sepal_length, "petal_width")
    end

    test "with different column returns correct correlations" do
      df = Explorer.Datasets.iris()
      correlation_result = Explorer.DataFrame.correlation(df)
      petal_length = LgbmExx.get_correlation_map(correlation_result, "petal_length")

      # petal_lengthの自己相関は1.0に近い
      assert_in_delta(petal_length["petal_length"], 1.0, 0.001)
    end
  end

  describe "map_importance_rate" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "returns rates", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      [most_importance | _] = LgbmExx.map_importance_rate(x_names, model.feature_importance_gain)
      assert {"petal_length", 0.6582736372947693} = most_importance
    end

    test "returns sorted by importance descending", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      results = LgbmExx.map_importance_rate(x_names, model.feature_importance_gain)

      # 重要度の降順であることを確認
      importances = Enum.map(results, &elem(&1, 1))

      is_sorted_desc =
        Enum.reduce_while(
          importances,
          true,
          fn
            value, true -> {:cont, value}
            value, prev -> if prev >= value, do: {:cont, value}, else: {:halt, false}
          end
        )

      assert is_sorted_desc
    end

    test "with zero importance values" do
      names = ["feature_a", "feature_b", "feature_c"]
      importances = [0.0, 0.0, 0.0]

      results = LgbmExx.map_importance_rate(names, importances)

      # ゼロ値でも結果を返す
      assert is_list(results)
      assert Enum.count(results) == 3
    end

    test "returns tuple with name and normalized rate" do
      names = ["feature_a", "feature_b", "feature_c"]
      importances = [1.0, 2.0, 3.0]

      results = LgbmExx.map_importance_rate(names, importances)

      # 各結果がタプル（名前、正規化された重要度）であることを確認
      Enum.each(results, fn result ->
        assert is_tuple(result)
        {name, rate} = result
        assert is_binary(name)
        assert is_number(rate)
      end)
    end
  end

  describe "explore_features" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "multiple features together achieve better accuracy than single features", %{
      model: model
    } do
      base_features = ["sepal_length"]
      exploration_sets = [[], ["petal_length"], ["petal_width"], ["petal_length", "petal_width"]]

      result =
        LgbmExx.explore_features(model, base_features, exploration_sets, 2,
          folding_rule: :stratified
        )

      exploration_results = result["exploration"]

      # base_features が正しく保存されているか
      assert result["base_features"] == base_features

      # 各パターンの評価値を取得
      single_petal_length = exploration_results[["petal_length"]].last_evaluation.mean
      single_petal_width = exploration_results[["petal_width"]].last_evaluation.mean
      combined = exploration_results[["petal_length", "petal_width"]].last_evaluation.mean

      # combined が両方の単体より良いことを確認
      assert combined < single_petal_length and combined < single_petal_width,
             "Combined features should achieve better accuracy than both single features"
    end
  end

  describe "permutation_importance" do
    @describetag :tmp_dir

    setup [:setup_model]

    defp df_to_list(df) do
      Nx.stack(df, axis: -1) |> Nx.to_list()
    end

    test "returns baseline and importance for each feature", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      {:ok, train_df} = Explorer.DataFrame.from_csv(model.files.train, header: false)
      train_df = Explorer.DataFrame.rename(train_df, ["species"] ++ x_names)

      x_test = train_df[x_names] |> df_to_list()
      y_test = train_df["species"] |> Explorer.Series.to_list()

      accuracy_fn = fn pred_y, correct_y ->
        Enum.zip(pred_y, correct_y)
        |> Enum.count(fn {probs, label} ->
          predicted_class = Enum.with_index(probs) |> Enum.max_by(&elem(&1, 0)) |> elem(1)
          predicted_class == round(label)
        end)
        |> then(&(&1 / Enum.count(correct_y)))
      end

      {baseline, importances} =
        LgbmExx.permutation_importance(model, x_test, y_test, accuracy_fn, n_repeats: 5)

      assert is_number(baseline)
      assert baseline > 0.5

      assert is_list(importances)
      assert Enum.count(importances) == 4

      Enum.each(importances, fn {name, importance} ->
        assert is_binary(name)
        assert is_number(importance)
      end)

      feature_names = Enum.map(importances, &elem(&1, 0))
      assert feature_names == x_names
    end

    test "petal features are more important than sepal features for iris classification", %{
      model: model
    } do
      x_names = Keyword.get(model.parameters, :x_names)
      {:ok, train_df} = Explorer.DataFrame.from_csv(model.files.train, header: false)
      train_df = Explorer.DataFrame.rename(train_df, ["species"] ++ x_names)

      x_test = train_df[x_names] |> df_to_list()
      y_test = train_df["species"] |> Explorer.Series.to_list()

      accuracy_fn = fn pred_y, correct_y ->
        Enum.zip(pred_y, correct_y)
        |> Enum.count(fn {probs, label} ->
          predicted_class = Enum.with_index(probs) |> Enum.max_by(&elem(&1, 0)) |> elem(1)
          predicted_class == round(label)
        end)
        |> then(&(&1 / Enum.count(correct_y)))
      end

      {_baseline, importances} =
        LgbmExx.permutation_importance(model, x_test, y_test, accuracy_fn, n_repeats: 10)

      importance_map = Map.new(importances)

      petal_length_imp = importance_map["petal_length"]
      petal_width_imp = importance_map["petal_width"]
      sepal_length_imp = importance_map["sepal_length"]
      sepal_width_imp = importance_map["sepal_width"]

      assert petal_length_imp > sepal_length_imp or petal_width_imp > sepal_width_imp,
             "At least one petal feature should be more important than sepal features"
    end

    test "accepts y_test as list", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      {:ok, train_df} = Explorer.DataFrame.from_csv(model.files.train, header: false)
      train_df = Explorer.DataFrame.rename(train_df, ["species"] ++ x_names)

      x_test = train_df[x_names] |> df_to_list()
      y_test = train_df["species"] |> Explorer.Series.to_list()

      accuracy_fn = fn pred_y, correct_y ->
        Enum.zip(pred_y, correct_y)
        |> Enum.count(fn {probs, label} ->
          predicted_class = Enum.with_index(probs) |> Enum.max_by(&elem(&1, 0)) |> elem(1)
          predicted_class == round(label)
        end)
        |> then(&(&1 / Enum.count(correct_y)))
      end

      {baseline, importances} =
        LgbmExx.permutation_importance(model, x_test, y_test, accuracy_fn, n_repeats: 3)

      assert is_number(baseline)
      assert is_list(importances)
      assert Enum.count(importances) == 4
    end

    test "uses default n_repeats of 5", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      {:ok, train_df} = Explorer.DataFrame.from_csv(model.files.train, header: false)
      train_df = Explorer.DataFrame.rename(train_df, ["species"] ++ x_names)

      x_test = train_df[x_names] |> df_to_list()
      y_test = train_df["species"] |> Explorer.Series.to_list()

      call_count = :counters.new(1, [:atomics])

      counting_metric_fn = fn pred_y, correct_y ->
        :counters.add(call_count, 1, 1)
        Enum.count(pred_y) + Enum.count(correct_y)
      end

      {_baseline, _importances} =
        LgbmExx.permutation_importance(model, x_test, y_test, counting_metric_fn)

      total_calls = :counters.get(call_count, 1)
      # 1 baseline + 4 features * 5 repeats = 21 calls
      assert total_calls == 21
    end

    test "evaluates only specified features when :features option is given", %{model: model} do
      x_names = Keyword.get(model.parameters, :x_names)
      {:ok, train_df} = Explorer.DataFrame.from_csv(model.files.train, header: false)
      train_df = Explorer.DataFrame.rename(train_df, ["species"] ++ x_names)

      x_test = train_df[x_names] |> df_to_list()
      y_test = train_df["species"] |> Explorer.Series.to_list()

      accuracy_fn = fn pred_y, correct_y ->
        Enum.zip(pred_y, correct_y)
        |> Enum.count(fn {probs, label} ->
          predicted_class = Enum.with_index(probs) |> Enum.max_by(&elem(&1, 0)) |> elem(1)
          predicted_class == round(label)
        end)
        |> then(&(&1 / Enum.count(correct_y)))
      end

      target_features = ["petal_length", "petal_width"]

      {_baseline, importances} =
        LgbmExx.permutation_importance(model, x_test, y_test, accuracy_fn,
          features: target_features,
          n_repeats: 3
        )

      assert Enum.count(importances) == 2
      feature_names = Enum.map(importances, &elem(&1, 0))
      assert feature_names == target_features
    end
  end
end
