defmodule LgbmExx.Splitter do
  @moduledoc """
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  require Explorer.DataFrame

  def split_train_data(model, k, folding_rule) do
    {:ok, train_df} = read_csv(model.files.train)
    nx = Nx.stack(train_df, axis: -1)

    [{train_one, val_one} | _] = list = split(nx, k, folding_rule, model.num_classes)
    names = list_names(model)

    list_df =
      Enum.map(list, fn {train_nx, val_nx} ->
        {to_df(train_nx, names), to_df(val_nx, names)}
      end)

    {Nx.axis_size(train_one, 0), Nx.axis_size(val_one, 0), list_df}
  end

  defp split(nx, k, :raw, _) do
    _split(nx, k)
  end

  defp split(nx, k, :shuffle, _) do
    shuffle(nx) |> _split(k)
  end

  defp split(nx, k, :stratified, num_classes) do
    # 1. sort by prediction target value
    #   prediction target value is index: 0
    indexes = list_sorted_indexes(nx)

    # 2. divide data to k groups in order
    num_groups = if(num_classes, do: k * num_classes, else: k)
    indexes_groups = chunk_group(indexes, num_groups)

    # 3. concat k_fold_splitted data of each groups
    k_fold_split_on_groups(indexes_groups, k)
    |> Enum.map(fn {train_index_nx, val_index_nx} ->
      train_nx = Nx.take(nx, train_index_nx)
      val_nx = Nx.take(nx, val_index_nx)
      {train_nx, val_nx}
    end)
  end

  defp split(nx, k, :stratified_shuffle, num_classes) do
    # 1. sort by prediction target value
    #   prediction target value is index: 0
    indexes = list_sorted_indexes(nx)

    # 2. divide data to k groups in order with shuffle data in group
    num_groups = if(num_classes, do: k * num_classes, else: k)
    indexes_groups = chunk_group(indexes, num_groups, shuffle: true)

    # 3. concat k_fold_splitted data of each groups
    k_fold_split_on_groups(indexes_groups, k)
    |> Enum.map(fn {train_index_nx, val_index_nx} ->
      train_nx = Nx.take(nx, train_index_nx)
      val_nx = Nx.take(nx, val_index_nx)
      {train_nx, val_nx}
    end)
  end

  defp _split(nx, k) do
    Scholar.ModelSelection.k_fold_split(nx, k)
    |> Enum.to_list()
  end

  defp list_sorted_indexes(df) do
    Nx.take(df, Nx.tensor([0]), axis: 1)
    |> DataFrame.new()
    |> DataFrame.rename(["value"])
    |> DataFrame.mutate(index: Series.row_index(value))
    |> DataFrame.sort_by(asc: value)
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
    |> Enum.map(&(Series.to_tensor(&1) |> Scholar.ModelSelection.k_fold_split(k)))
    |> Enum.zip_reduce([], &(&2 ++ [&1]))
    |> Enum.map(fn group_parts ->
      # concat each group train/val tensor
      group_parts
      |> Enum.reduce({nil, nil}, fn
        {train_nx, val_nx}, {nil, nil} ->
          {train_nx, val_nx}

        {train_nx, val_nx}, {acc_train_nx, acc_val_nx} ->
          {
            Nx.concatenate([acc_train_nx, train_nx], axis: 0),
            Nx.concatenate([acc_val_nx, val_nx], axis: 0)
          }
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
