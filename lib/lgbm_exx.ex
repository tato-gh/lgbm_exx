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

  @doc """
  Explore feature combinations by adding exploration sets to base features.

  Generates all combinations of exploration_sets and evaluates each combination
  with base_features using cross-validation. Returns results organized by
  exploration set combinations.

  - uses "_cv" named model directory

  ## Args

  `model`: LightGBM model with training data
  `base_features`: List of feature names always included [e.g., ["a", "b"]]
  `exploration_sets`: List of feature sets to explore [e.g., [["c"], ["d"], ["c", "d"]]]
  `k`: Number of folds for cross-validation
  `options`: Same as cross_validate/3 options

  ## Returns

  ```
  %{
    "base_features" => ["a", "b"],
    "exploration" => %{
      [] => aggregated_result_0,
      ["c"] => aggregated_result_1,
      ["d"] => aggregated_result_2,
      ["c", "d"] => aggregated_result_3
    }
  }
  ```
  """
  def explore_features(model, base_features, exploration_sets, k, options \\ []) do
    # Get original model parameters
    y_name = Keyword.get(model.parameters, :y_name)
    model_name = model.name

    # Read original train and validate data
    {:ok, train_df_raw} = DF.from_csv(model.files.train, header: false)
    {:ok, val_df_raw} = DF.from_csv(model.files.validation, header: false)

    # Get all column names from the model's original parameters
    all_cols = [y_name] ++ Keyword.get(model.parameters, :x_names)

    # Rename columns to match the model's expectations
    train_df = DF.rename(train_df_raw, all_cols)
    val_df = DF.rename(val_df_raw, all_cols)

    exploration_results =
      exploration_sets
      |> Enum.reduce(%{}, fn exploration_set, acc ->
        actual_features = base_features ++ exploration_set

        # Select only the features we need (y_name + actual_features)
        selected_cols = [y_name] ++ actual_features
        train_df_subset = DF.select(train_df, selected_cols)
        val_df_subset = DF.select(val_df, selected_cols)

        # Create new model with reduced feature set
        model_tmp =
          LgbmEx.fit(
            "#{model_name}_exploration",
            {train_df_subset, val_df_subset},
            y_name,
            model.parameters
          )

        {_, _, cv_results} = cross_validate(model_tmp, k, options)
        aggregated = aggregate_cv_results(cv_results)
        x_names = Keyword.get(model_tmp.parameters, :x_names)
        aggregated_with_names = Map.put(aggregated, :x_names, x_names)

        Map.put(acc, exploration_set, aggregated_with_names)
      end)

    %{
      "base_features" => base_features,
      "exploration" => exploration_results
    }
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
    # Explorer.DataFrame.dummies/2 generates a column with "{col_name}_" suffix
    # for each original column containing nil values. We exclude these nil-representing
    # columns from the statistics calculation to avoid counting nil values in the results.
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
    describe_labels = stats_map["describe"]

    stats_map
    |> Map.delete("describe")
    |> Enum.reduce(%{}, fn {column, values}, acc ->
      stats = Enum.zip(describe_labels, values) |> Map.new()
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

  @doc """
  Compute permutation importance for a trained model.

  Permutation importance measures how much the model performance decreases
  when a single feature's values are randomly shuffled.

  ## Args

  - `model`: LgbmEx.Model - trained model
  - `x_test`: Explorer.DataFrame - test feature data with column names matching model.parameters[:x_names]
  - `y_test`: Explorer.Series or List - true labels
  - `metric_fn`: Function - evaluation function `fn(pred_y :: list, correct_y :: list) -> scalar`
    - Higher is better metrics (e.g., accuracy): positive importance means important feature
    - Lower is better metrics (e.g., MSE): negative importance means important feature

  ## Options

  - `n_repeats`: number of times to permute a feature (default: 20)

  ## Returns

  `{baseline_score, [{feature_name :: String, importance_value :: Number}, ...]}`

  Returns tuple of baseline score and list of importances in original feature order.
  """
  def permutation_importance(model, x_test, y_test, metric_fn, opts \\ []) do
    n_repeats = Keyword.get(opts, :n_repeats, 20)

    y_test_list = normalize_to_list(y_test)
    x_names = Keyword.get(model.parameters, :x_names)

    baseline_pred = LgbmEx.predict(model, x_test)
    baseline_score = metric_fn.(baseline_pred, y_test_list)

    importances =
      x_names
      |> Enum.with_index()
      |> Enum.map(fn {feature_name, col_idx} ->
        shuffled_scores =
          for _ <- 1..n_repeats do
            shuffled_x_test = shuffle_column_at(x_test, col_idx)
            shuffled_pred = LgbmEx.predict(model, shuffled_x_test)
            metric_fn.(shuffled_pred, y_test_list)
          end

        avg_shuffled_score = Enum.sum(shuffled_scores) / n_repeats
        importance = baseline_score - avg_shuffled_score

        {feature_name, importance}
      end)

    {baseline_score, importances}
  end

  defp normalize_to_list(%Explorer.Series{} = series), do: Explorer.Series.to_list(series)
  defp normalize_to_list(list) when is_list(list), do: list

  defp shuffle_column_at(rows, col_idx) do
    col_values = Enum.map(rows, &Enum.at(&1, col_idx))
    shuffled_col = Enum.shuffle(col_values)

    rows
    |> Enum.zip(shuffled_col)
    |> Enum.map(fn {row, new_val} -> List.replace_at(row, col_idx, new_val) end)
  end
end
