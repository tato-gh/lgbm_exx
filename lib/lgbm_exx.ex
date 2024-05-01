defmodule LgbmExx do
  @moduledoc """
  Documentation for `LgbmExx`.
  """

  alias LgbmExx.Splitter
  alias LgbmExx.CVModel
  alias LgbmExx.CVResult
  alias Explorer.DataFrame, as: DF

  @doc """
  Returns evaluation values each k-folding model.

  Concats model train data and validation data to sample all data.

  - uses "_cv" named model directroy

  ## Args

  `k` is folding size.

  #### options

  `folding_rule` is one of `[:raw, :shuffle, :stratified, :stratified_shuffle]`

  - raw: folds data as is row number order (default)
  - shuffle: folds after shuffling data
  - stratified: folds after sort to obtain unbiased data in each group
  - stratified_shuffle: same as `stratified`, however shuffle first.

  `x_test` are test data. If `x_test` are specified, these predicted results of each cv-model are being included in results.

  `evaluator` is function to calculate evaluation value between val_data and those predicted value.
  """
  def cross_validate(model, k, options \\ []) do
    folding_rule = Keyword.get(options, :folding_rule, :raw)
    x_test = Keyword.get(options, :x_test, [])
    evaluator = Keyword.get(options, :evaluator, nil)

    {train_size, val_size, list} = Splitter.split_train_data(model, k, folding_rule)

    results =
      Enum.map(list, fn {train_df, val_df} ->
        CVModel.fit_and_evaluate(model, {train_df, val_df}, x_test, evaluator)
      end)

    {train_size, val_size, results}
  end

  def aggregate_cv_results(cv_results) do
    CVResult.aggregate(cv_results)
  end

  @doc """
  Returns cross validation results of grid (parameters list).

  - uses "_grid" named model directroy

  ## Args

  `grid` is like below.

  ```
  grid = [
    num_iterations: [5, 10],
    min_data_in_leaf: [2, 3]
  ]
  ```

  `k` and `cv_options` is cross validation args see `cross_validation/3`.
  """
  def grid_search(model, grid, k, cv_options \\ []) do
    model_grid = LgbmEx.copy_model(model, "_grid")

    combinations(grid)
    |> Enum.map(fn parameters ->
      model_tmp = LgbmEx.refit_model(model_grid, parameters)
      {_, _, cv_results} = cross_validate(model_tmp, k, cv_options)

      {parameters, aggregate_cv_results(cv_results)}
    end)
  end

  defp combinations([]), do: [[]]

  defp combinations([{name, values} | rest]) do
    for sub <- combinations(rest), value <- values do
      [{name, value} | sub]
    end
  end

  @doc """
  One-hot encoding

  ## Args

  df: DataFrame
  columns: target columns. The columns are not incluced in the returns.
  threshold: Only elements exceeding this number will be considered.
  """
  def one_hot_encode(df, columns, threshold \\ 3)

  def one_hot_encode(df, [], _threshold), do: df

  def one_hot_encode(df, nil, _threshold), do: df

  def one_hot_encode(df, columns, threshold) do
    # HACKME
    nil_names = Enum.map(columns, &(&1 <> "_"))
    dummies = DF.dummies(df, columns)
    one_hot_names = DF.names(dummies)
    one_hot_names_without_nil = one_hot_names -- nil_names

    columns_stats(dummies, one_hot_names_without_nil)
    |> Enum.filter(fn {_name, %{"count" => count}} -> count > threshold end)
    |> Enum.map(&elem(&1, 0))
    |> case do
      [] ->
        DF.discard(df, columns)

      valid_one_hot_names ->
        dummies = DF.select(dummies, valid_one_hot_names)

        df
        |> DF.concat_columns(dummies)
        |> DF.discard(columns)
    end
  end

  @doc """
  Returns columns stats as map.
  """
  def columns_stats(df, columns) do
    stats_map = DF.describe(df[columns]) |> DF.to_columns()

    Map.delete(stats_map, "describe")
    |> Enum.reduce(%{}, fn {column, values}, acc ->
      stats = Enum.zip(stats_map["describe"], values) |> Map.new()
      Map.put(acc, column, stats)
    end)
  end

  @doc """
  Get given column map from DataFrame.correlation results.

  ## Args

  df: DataFrame.correlation result.
  column: target column name.
  """
  def get_correlation_map(df, column) do
    all = DF.to_columns(df)
    column_index = Enum.find_index(all["names"], &(&1 == column))

    Enum.reduce(all["names"], %{}, fn name, acc ->
      value = Enum.at(all[name], column_index)
      Map.put(acc, name, value)
    end)
  end

  @doc """
  Convert feature_importance to rate
  """
  def map_importance_rate(names, feature_importance) do
    l1 =
      feature_importance
      |> Nx.tensor()
      |> Scholar.Preprocessing.normalize(norm: :manhattan)
      |> Nx.to_list()

    Enum.zip(names, l1)
    |> Enum.sort_by(&elem(&1, 1), :desc)
  end
end
