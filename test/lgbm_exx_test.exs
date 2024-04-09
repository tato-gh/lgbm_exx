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

      {_, _, [result | _]} = LgbmExx.cross_validate(model, 3, x_test: x_test)
      [[p1, _, _], [_, p2, _], [_, _, p3]] = result.prediction

      assert p1 > 0.5
      assert p2 > 0.5
      assert p3 > 0.5
    end

    test "uses with folding_rule", %{model: model} do
      result =
        [:raw, :shuffle, :sort, :sort_with_shuffle]
        |> Map.new(fn rule ->
          {_, _, [result | _]} = LgbmExx.cross_validate(model, 3, folding_rule: rule)
          {rule, result}
        end)

      assert result.raw.last_evaluation >= 1.0
      assert result.shuffle.last_evaluation < 0.3
      assert result.sort.last_evaluation < 0.3
      assert result.sort_with_shuffle.last_evaluation < 0.3
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

      assert result.evaluator_result >= 0.5
    end
  end

  describe "aggregate_cv_results" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "returns aggregation result", %{model: model} do
      {100, 50, cv_results} = LgbmExx.cross_validate(model, 3)

      last_evaluation =
        Enum.map(cv_results, & &1.last_evaluation)
        |> Explorer.Series.from_list()
        |> Explorer.Series.mean()

      result = LgbmExx.aggregate_cv_results(cv_results)

      assert result.last_evaluation == last_evaluation
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
      assert result.num_iterations == 5
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
  end

  describe "columns_stats" do
    @describetag :tmp_dir

    test "returns statistics map" do
      df = Explorer.Datasets.iris()
      stats = LgbmExx.columns_stats(df, ["sepal_length", "sepal_width"])

      assert stats["sepal_length"]["count"] == 150
      assert stats["sepal_width"]["count"] == 150
    end
  end
end
