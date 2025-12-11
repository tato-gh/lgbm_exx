defmodule LgbmExx.CVResult do
  @moduledoc """
  """

  @aggregation_targets ~w(
    num_iterations
    last_evaluation
    evaluator_result
    prediction
    feature_importance_split
    feature_importance_gain
  )a

  def aggregate(cv_results) do
    cv_results
    |> Enum.filter(& &1)
    |> Enum.map(fn result ->
      Enum.map(@aggregation_targets, &Map.get(result, &1))
    end)
    |> Enum.zip_reduce([], &(&2 ++ [&1]))
    |> Enum.zip(@aggregation_targets)
    |> Enum.map(&get_stats/1)
    |> Enum.filter(& &1)
    |> Map.new()
  end

  defp get_stats({values, :evaluator_result}) do
    # check values list or tuples list
    elem = List.first(values)

    cond do
      is_tuple(elem) ->
        values
        |> Enum.map(&Tuple.to_list/1)
        |> Enum.zip_reduce([], &(&2 ++ [&1]))
        |> Enum.map(&calc_stats/1)
        |> then(&{:evaluator_result, &1})

      true ->
        {:evaluator_result, calc_stats(Enum.filter(values, & &1))}
    end
  end

  defp get_stats({values, key})
       when key in [:num_iterations, :last_evaluation] do
    {key, calc_stats(Enum.filter(values, & &1))}
  end

  defp get_stats({values, key})
       when key in [:feature_importance_split, :feature_importance_gain] do
    # k回分を統合してそれぞれ返す
    result =
      values
      |> Enum.zip_reduce([], &(&2 ++ [&1]))
      |> Enum.map(&calc_mean/1)

    {key, result}
  end

  defp get_stats({values, :prediction}) do
    # Aggregate predictions from k-fold cross-validation:
    # 1. Transpose k predictions (one per fold) to group by row
    # 2. For each row, calculate mean prediction across folds
    prediction =
      values
      |> transpose_predictions()
      |> Enum.map(&average_row_predictions/1)

    {:prediction, prediction}
  end

  defp get_stats(_), do: nil

  defp transpose_predictions(predictions) do
    # Convert list of k predictions into rows grouped by index
    predictions
    |> Enum.zip_reduce([], &(&2 ++ [&1]))
  end

  defp average_row_predictions(row_probs_list) do
    # For each row, average the prediction values across k folds
    row_probs_list
    |> Enum.zip_reduce([], &(&2 ++ [&1]))
    |> Enum.map(&calc_mean/1)
  end

  defp calc_mean([]), do: nil

  defp calc_mean(values) do
    values
    |> Explorer.Series.from_list()
    |> Explorer.Series.mean()
  end

  defp calc_stats([]), do: nil

  defp calc_stats(values) do
    s = values |> Explorer.Series.from_list()

    %{
      mean: Explorer.Series.mean(s),
      med: Explorer.Series.median(s),
      std: Explorer.Series.standard_deviation(s),
      max: Explorer.Series.max(s),
      min: Explorer.Series.min(s)
    }
  end
end
