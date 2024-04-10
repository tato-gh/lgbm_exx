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

  defp get_stats({values, key})
       when key in [:num_iterations, :last_evaluation, :evaluator_result] do
    {key, calc_mean(Enum.filter(values, & &1))}
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
    prediction =
      values
      |> Enum.zip_reduce([], &(&2 ++ [&1]))
      |> Enum.map(fn row_probs_list ->
        Enum.zip_reduce(row_probs_list, [], &(&2 ++ [&1]))
        |> Enum.map(&calc_mean/1)
      end)

    {:prediction, prediction}
  end

  defp get_stats(_), do: nil

  defp calc_mean([]), do: nil

  defp calc_mean(values) do
    size = Enum.count(values)
    sum = Enum.sum(values)

    sum / size
  end
end
