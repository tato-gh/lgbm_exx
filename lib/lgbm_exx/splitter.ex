defmodule LgbmExx.Splitter do
  @moduledoc """
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  require Explorer.DataFrame

  def split_train_data(model, k, folding_rule) do
    {:ok, train_df} = read_csv(model.files.train)
    {:ok, val_df} = read_validation_csv(model.files.train, model.files.validation)
    full_nx = concat_rows(train_df, val_df) |> Nx.stack(axis: -1)

    [{train_one, val_one} | _] = list = split(full_nx, k, folding_rule)
    names = list_names(model)

    list_df =
      Enum.map(list, fn {train_nx, val_nx} ->
        {to_df(train_nx, names), to_df(val_nx, names)}
      end)

    {Nx.axis_size(train_one, 0), Nx.axis_size(val_one, 0), list_df}
  end

  defp split(full_nx, k, :raw) do
    _split(full_nx, k)
  end

  defp split(full_nx, k, :shuffle) do
    shuffle(full_nx) |> _split(k)
  end

  defp split(full_nx, k, :sort) do
    # 1. sort by prediction target value
    #   prediction target value is index: 0
    indexes = list_sorted_indexes(full_nx)

    # 2. divide data to k groups in order
    indexes_groups = chunk_group(indexes, k)

    # 3. concat k_fold_splitted data of each groups
    k_fold_split_on_groups(indexes_groups, k)
    |> Enum.map(fn {train_index_nx, val_index_nx} ->
      train_nx = Nx.take(full_nx, train_index_nx)
      val_nx = Nx.take(full_nx, val_index_nx)
      {train_nx, val_nx}
    end)
  end

  defp split(full_nx, k, :sort_with_shuffle) do
    # 1. sort by prediction target value
    #   prediction target value is index: 0
    indexes = list_sorted_indexes(full_nx)

    # 2. divide data to k groups in order with shuffle data in group
    indexes_groups = chunk_group(indexes, k, shuffle: true)

    # 3. concat k_fold_splitted data of each groups
    k_fold_split_on_groups(indexes_groups, k)
    |> Enum.map(fn {train_index_nx, val_index_nx} ->
      train_nx = Nx.take(full_nx, train_index_nx)
      val_nx = Nx.take(full_nx, val_index_nx)
      {train_nx, val_nx}
    end)
  end

  defp _split(full_nx, k) do
    Scholar.ModelSelection.k_fold_split(full_nx, k)
    |> Enum.to_list()
  end

  defp list_sorted_indexes(df) do
    Nx.take(df, Nx.tensor([0]), axis: 1)
    |> DataFrame.new()
    |> DataFrame.rename(["value"])
    |> DataFrame.mutate(index: Series.row_index(value))
    |> DataFrame.sort_by(value)
    |> DataFrame.pull("index")
  end

  defp chunk_group(series, k, options \\ []) do
    shuffle = Keyword.get(options, :shuffle, false)
    group_size = round(Series.count(series) / k)

    Enum.map(1..k, fn no ->
      offset = (no - 1) * group_size

      Series.slice(series, offset, group_size)
      |> maybe_shuffle_series(shuffle)
    end)
  end

  defp k_fold_split_on_groups(groups, k) do
    groups
    |> Enum.reduce([], fn group, acc ->
      Series.to_tensor(group)
      |> Scholar.ModelSelection.k_fold_split(k)
      |> Enum.to_list()
      |> Enum.with_index(0)
      |> Enum.reduce(acc, fn {{train_nx, val_nx}, i_k}, acc_i ->
        Enum.at(acc_i, i_k)
        |> if do
          List.update_at(acc_i, i_k, fn {train_acc, val_acc} ->
            {
              Nx.concatenate([train_acc, train_nx], axis: 0),
              Nx.concatenate([val_acc, val_nx], axis: 0)
            }
          end)
        else
          List.insert_at(acc_i, i_k, {train_nx, val_nx})
        end
      end)
    end)
  end

  defp shuffle(tensor) do
    key = Nx.Random.key(:rand.uniform(100))
    {shuffled, _new_key} = Nx.Random.shuffle(key, tensor, axis: 0)

    shuffled
  end

  defp maybe_shuffle_series(series, true), do: Series.shuffle(series)

  defp maybe_shuffle_series(series, _false), do: series

  defp read_csv(path) do
    File.exists?(path)
    |> if(do: DataFrame.from_csv(path, header: false))
  end

  defp read_validation_csv(train_path, validation_path) do
    # skip if validation_csv is the hard copy file (that means data is same)
    read? =
      File.exists?(validation_path) &&
        File.stat!(train_path).inode != File.stat!(validation_path).inode

    if(read?, do: DataFrame.from_csv(validation_path, header: false), else: {:ok, nil})
  end

  defp concat_rows(train_df, nil), do: train_df

  defp concat_rows(train_df, val_df) do
    DataFrame.concat_rows(train_df, val_df)
  end

  defp list_names(model) do
    y_name = Keyword.get(model.parameters, :y_name)
    x_names = Keyword.get(model.parameters, :x_names)
    [y_name] ++ x_names
  end

  defp to_df(tensor, names) do
    tensor
    |> DataFrame.new()
    |> DataFrame.rename(names)
  end
end
