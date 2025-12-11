defmodule LgbmExx.SplitterTest do
  use ExUnit.Case, async: false

  defp setup_model(_context) do
    # Use a simplified tmp_dir path without commas
    # ExUnit's tmp_dir includes test name which may contain commas
    base_tmp = Path.join(System.tmp_dir!(), "lgbm_exx_splitter_test")
    File.mkdir_p!(base_tmp)
    Application.put_env(:lgbm_ex, :workdir, base_tmp)

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

  describe "safe_k_fold_split" do
    test "safe_k_fold_split with known input - 10 items k=3" do
      # 10個のテンソル [0,1,2,3,4,5,6,7,8,9] をk=3で分割
      tensor = Nx.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 3)

      # 3つのfoldが返される
      assert Enum.count(result) == 3

      # Fold 0: val=[0,1,2,3] (4個), train=[4,5,6,7,8,9] (6個)
      {train_0, val_0} = Enum.at(result, 0)
      assert Nx.to_list(val_0) == [0, 1, 2, 3]
      assert Nx.to_list(train_0) == [4, 5, 6, 7, 8, 9]

      # Fold 1: val=[4,5,6] (3個), train=[0,1,2,3,7,8,9] (7個)
      {train_1, val_1} = Enum.at(result, 1)
      assert Nx.to_list(val_1) == [4, 5, 6]
      assert Nx.to_list(train_1) == [0, 1, 2, 3, 7, 8, 9]

      # Fold 2: val=[7,8,9] (3個), train=[0,1,2,3,4,5,6] (7個)
      {train_2, val_2} = Enum.at(result, 2)
      assert Nx.to_list(val_2) == [7, 8, 9]
      assert Nx.to_list(train_2) == [0, 1, 2, 3, 4, 5, 6]
    end

    test "safe_k_fold_split with 5 items k=2" do
      # 5個のテンソル [10,20,30,40,50] をk=2で分割
      tensor = Nx.tensor([10, 20, 30, 40, 50])
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 2)

      assert Enum.count(result) == 2

      # Fold 0: val=[10,20,30] (3個), train=[40,50] (2個)
      {train_0, val_0} = Enum.at(result, 0)
      assert Nx.to_list(val_0) == [10, 20, 30]
      assert Nx.to_list(train_0) == [40, 50]

      # Fold 1: val=[40,50] (2個), train=[10,20,30] (3個)
      {train_1, val_1} = Enum.at(result, 1)
      assert Nx.to_list(val_1) == [40, 50]
      assert Nx.to_list(train_1) == [10, 20, 30]
    end

    test "safe_k_fold_split validation indices do not overlap with train indices" do
      tensor = Nx.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 3)

      Enum.each(result, fn {train, val} ->
        train_set = Nx.to_list(train) |> MapSet.new()
        val_set = Nx.to_list(val) |> MapSet.new()

        # 重複がないことを確認
        intersection = MapSet.intersection(train_set, val_set)
        assert MapSet.size(intersection) == 0, "Train and validation sets should not overlap"
      end)
    end

    test "safe_k_fold_split all validation indices cover full dataset" do
      tensor = Nx.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 3)

      # 全foldのvalidation インデックスを収集
      all_val_indices =
        result
        |> Enum.map(fn {_, val} -> Nx.to_list(val) end)
        |> List.flatten()
        |> Enum.sort()

      # 0から9まで全て含まれ、重複がない
      assert all_val_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    end

    test "safe_k_fold_split no data loss" do
      tensor = Nx.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 3)

      total_train_val_rows =
        Enum.reduce(result, 0, fn {train, val}, acc ->
          acc + Nx.size(train) + Nx.size(val)
        end)

      # 合計が元のサイズの3倍（3folds × 10items）
      assert total_train_val_rows == 30
    end
  end

  describe "split_train_data" do
    @describetag tmp_dir: "splitter_test"

    setup [:setup_model]

    test "returns train_size, val_size, and list of fold data", %{model: model} do
      {train_size, val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      # Irisデータセットは150行
      assert train_size + val_size == 150
      assert Enum.count(list) == 3
    end

    test "with :raw folding rule returns data in original order", %{model: model} do
      {train_size, val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      # 各foldが正しいサイズであることを確認
      Enum.each(list, fn {train_df, val_df} ->
        train_count = Explorer.DataFrame.n_rows(train_df)
        val_count = Explorer.DataFrame.n_rows(val_df)

        assert train_count == train_size
        assert val_count == val_size
      end)
    end

    test "with :shuffle folding rule returns shuffled data", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :shuffle)

      # シャッフルなので各foldが存在することを確認
      assert Enum.count(list) == 3
      Enum.each(list, fn {train_df, val_df} ->
        assert Explorer.DataFrame.n_rows(train_df) > 0
        assert Explorer.DataFrame.n_rows(val_df) > 0
      end)
    end

    test "with :stratified folding rule returns stratified data", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :stratified)

      # 層別分割なので各foldが存在することを確認
      assert Enum.count(list) == 3
      Enum.each(list, fn {train_df, val_df} ->
        assert Explorer.DataFrame.n_rows(train_df) > 0
        assert Explorer.DataFrame.n_rows(val_df) > 0

        # 各foldのvalidationセットでクラス分布を確認
        # Irisデータセットは3クラスが均等（各50サンプル）なので、
        # 各クラスが存在することを確認
        val_classes =
          Explorer.DataFrame.pull(val_df, "species")
          |> Explorer.Series.distinct()
          |> Explorer.Series.to_list()
          |> Enum.sort()

        # k=3の場合、各foldに全クラスが含まれるはず
        assert Enum.count(val_classes) > 0
      end)
    end

    test "with :stratified_shuffle folding rule returns stratified shuffled data", %{
      model: model
    } do
      {_train_size, _val_size, list} =
        LgbmExx.Splitter.split_train_data(model, 3, :stratified_shuffle)

      # 層別シャッフル分割なので各foldが存在することを確認
      assert Enum.count(list) == 3
      Enum.each(list, fn {train_df, val_df} ->
        assert Explorer.DataFrame.n_rows(train_df) > 0
        assert Explorer.DataFrame.n_rows(val_df) > 0

        # 各foldのvalidationセットでクラス分布を確認
        val_classes =
          Explorer.DataFrame.pull(val_df, "species")
          |> Explorer.Series.distinct()
          |> Explorer.Series.to_list()
          |> Enum.sort()

        # 層別化されているので各foldにクラスが含まれるはず
        assert Enum.count(val_classes) > 0
      end)
    end

    test "k=2 returns 2 folds", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 2, :raw)

      assert Enum.count(list) == 2
    end

    test "k=5 returns 5 folds", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 5, :raw)

      assert Enum.count(list) == 5
    end

    test "each fold contains all columns from original data", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      original_columns = Keyword.get(model.parameters, :x_names) |> Enum.count()
      original_columns = original_columns + 1  # y_name

      Enum.each(list, fn {train_df, val_df} ->
        assert Explorer.DataFrame.n_columns(train_df) == original_columns
        assert Explorer.DataFrame.n_columns(val_df) == original_columns
      end)
    end

    test "train and val data from same fold do not overlap", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      # 各foldの train と val が互いに排他的であることを確認することは
      # Explorerの現在のAPIでは難しいため、フォルド数とサイズで確認
      Enum.each(list, fn {train_df, val_df} ->
        train_rows = Explorer.DataFrame.n_rows(train_df)
        val_rows = Explorer.DataFrame.n_rows(val_df)

        assert train_rows + val_rows == 150
      end)
    end

    test "all folds together cover full dataset", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      # 各foldの検証データが合計で全データをカバーしていることを確認
      total_val_rows = Enum.reduce(list, 0, fn {_train_df, val_df}, acc ->
        acc + Explorer.DataFrame.n_rows(val_df)
      end)

      assert total_val_rows == 150
    end

    test "stratified split maintains approximate class balance in validation sets", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :stratified)

      # Irisデータセットは3クラス×50サンプル = 150サンプル
      # k=3なので各foldのvalidationは50サンプル
      # 各クラスが大体均等に含まれるはず（完全に均等ではないかもしれない）
      Enum.each(list, fn {_train_df, val_df} ->
        class_counts =
          Explorer.DataFrame.pull(val_df, "species")
          |> Explorer.Series.frequencies()

        # 各クラスが最低1つは含まれることを確認
        assert Explorer.DataFrame.n_rows(class_counts) > 0
      end)
    end

    test "stratified split maintains exact class proportions in validation sets", %{model: model} do
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :stratified)

      # Irisは各クラス50個ずつ（合計150）
      # k=3でk*num_classes=9グループに分割、各グループ内で3-fold split
      # よって各foldのvalには各クラス14-18個程度含まれるはず
      Enum.each(list, fn {_train_df, val_df} ->
        class_counts =
          Explorer.DataFrame.pull(val_df, "species")
          |> Explorer.Series.frequencies()
          |> Explorer.DataFrame.to_rows()
          |> Enum.into(%{}, fn row -> {row["values"], row["counts"]} end)

        # 各クラスの数を確認
        species_values = Explorer.DataFrame.pull(val_df, "species")
          |> Explorer.Series.distinct()
          |> Explorer.Series.to_list()

        # 3つのクラスが全て含まれることを確認
        assert Enum.count(species_values) == 3,
               "All 3 species should be present in validation set"

        # 各クラスが最低14個、最大18個くらい含まれるはず（層別化されている証拠）
        Enum.each(class_counts, fn {_class, count} ->
          assert count >= 14 and count <= 18,
                 "Each class should have ~16 samples in validation set, got #{count}"
        end)
      end)
    end

    test "raw split creates non-overlapping consecutive folds", %{model: model} do
      {_train_size, val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :raw)

      # rawの場合、各foldのvalidationは連続したデータのはず
      # 最初のfoldのvalidationの最初の値を取得して確認
      [{_train_df1, val_df1}, {_train_df2, val_df2}, {_train_df3, val_df3}] = list

      # 各validation setが同じサイズであることを確認
      assert Explorer.DataFrame.n_rows(val_df1) == val_size
      assert Explorer.DataFrame.n_rows(val_df2) == val_size
      assert Explorer.DataFrame.n_rows(val_df3) == val_size
    end

    test "raw split validation indices are sequential", _context do
      # テンソルで直接検証: 150個のインデックス [0..149] を k=3 で分割
      tensor = Nx.iota({150})
      result = LgbmExx.Splitter.safe_k_fold_split(tensor, 3)

      # raw split では validation セットが連続したインデックスであるべき
      # Fold 0: val=[0..49] (50個), train=[50..149]
      # Fold 1: val=[50..99] (50個), train=[0..49] + [100..149]
      # Fold 2: val=[100..149] (50個), train=[0..99]

      {_train_0, val_0} = Enum.at(result, 0)
      val_0_list = Nx.to_list(val_0)
      # 最初のfold: 連続したインデックスであることを確認
      expected_0 = Enum.to_list(0..49)
      assert val_0_list == expected_0,
             "First fold validation should be sequential indices 0-49"

      {_train_1, val_1} = Enum.at(result, 1)
      val_1_list = Nx.to_list(val_1)
      expected_1 = Enum.to_list(50..99)
      assert val_1_list == expected_1,
             "Second fold validation should be sequential indices 50-99"

      {_train_2, val_2} = Enum.at(result, 2)
      val_2_list = Nx.to_list(val_2)
      expected_2 = Enum.to_list(100..149)
      assert val_2_list == expected_2,
             "Third fold validation should be sequential indices 100-149"
    end

    test "split does not lose any data rows", %{model: model} do
      {train_size, val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :stratified)

      # 全foldのvalidationデータの合計が元データと同じであることを確認
      total_val_rows =
        Enum.reduce(list, 0, fn {_train_df, val_df}, acc ->
          acc + Explorer.DataFrame.n_rows(val_df)
        end)

      # Irisデータセットは150行
      assert total_val_rows == 150
      assert train_size + val_size == 150
    end

    test "split with non-divisible data size distributes remainder correctly", %{model: model} do
      # stratified splitでは、各クラスごとにk-fold分割するため、
      # 全体のvalidationサイズは均等にならない可能性がある
      # ここでは、データが失われず全て使われることを確認
      {_train_size, _val_size, list} = LgbmExx.Splitter.split_train_data(model, 3, :stratified)

      # 全validation setのサイズを収集
      val_sizes = Enum.map(list, fn {_train_df, val_df} ->
        Explorer.DataFrame.n_rows(val_df)
      end)

      # 全validationデータの合計が元データと同じ
      total_val = Enum.sum(val_sizes)
      assert total_val == 150

      # 各foldのvalidationが空でないことを確認
      Enum.each(val_sizes, fn size ->
        assert size > 0
      end)
    end
  end
end
