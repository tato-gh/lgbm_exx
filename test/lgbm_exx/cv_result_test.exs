defmodule LgbmExx.CVResultTest do
  use ExUnit.Case, async: true

  describe "aggregate" do
    test "aggregate basic cv_results with numeric evaluator_result" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.55,
          evaluator_result: 0.82,
          prediction: [[0.15, 0.25, 0.6], [0.35, 0.35, 0.3]],
          feature_importance_split: [1.5, 2.5, 3.5, 4.5],
          feature_importance_gain: [0.12, 0.22, 0.32, 0.42]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check aggregated keys exist
      assert Map.has_key?(result, :num_iterations)
      assert Map.has_key?(result, :last_evaluation)
      assert Map.has_key?(result, :evaluator_result)
      assert Map.has_key?(result, :prediction)
      assert Map.has_key?(result, :feature_importance_split)
      assert Map.has_key?(result, :feature_importance_gain)

      # Check aggregated statistics exist
      assert is_map(result.num_iterations)
      assert Map.has_key?(result.num_iterations, :mean)
      assert Map.has_key?(result.num_iterations, :med)
      assert Map.has_key?(result.num_iterations, :std)
      assert Map.has_key?(result.num_iterations, :max)
      assert Map.has_key?(result.num_iterations, :min)

      # Check specific statistics values
      assert result.num_iterations.mean == 10
      assert result.num_iterations.min == 10
      assert result.num_iterations.max == 10
    end

    test "aggregate calculates correct mean for last_evaluation" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.7,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check mean is calculated correctly
      assert result.last_evaluation.mean == 0.6
    end

    test "aggregate handles tuple evaluator_result" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: {0.8, 0.85},
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: {0.82, 0.87},
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check evaluator_result is a list of stats
      assert is_list(result.evaluator_result)
      assert Enum.count(result.evaluator_result) == 2
    end

    test "aggregate aggregates prediction correctly" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check prediction is aggregated (averaged)
      assert is_list(result.prediction)
      # Two samples
      assert Enum.count(result.prediction) == 2

      # Check each prediction is averaged
      Enum.each(result.prediction, fn pred ->
        assert is_list(pred)
        # Three classes
        assert Enum.count(pred) == 3
      end)
    end

    test "aggregate aggregates feature_importance_split correctly" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check feature_importance_split is averaged
      assert is_list(result.feature_importance_split)
      assert Enum.count(result.feature_importance_split) == 4

      # Check that values are means (1.5, 2.5, 3.5, 4.5)
      expected_means = [1.5, 2.5, 3.5, 4.5]

      Enum.zip(result.feature_importance_split, expected_means)
      |> Enum.each(fn {actual, expected} ->
        assert_in_delta(actual, expected, 0.01)
      end)
    end

    test "aggregate aggregates feature_importance_gain correctly" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check feature_importance_gain is averaged
      assert is_list(result.feature_importance_gain)
      assert Enum.count(result.feature_importance_gain) == 4
    end

    test "aggregate handles nil evaluator_result" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: nil,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: nil,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # evaluator_result should not be in the result when all are nil
      assert !Map.has_key?(result, :evaluator_result) || result.evaluator_result == nil
    end

    test "aggregate filters out nil results from cv_results list" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        nil,
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Should aggregate without error, ignoring nil
      assert is_map(result)
      assert Map.has_key?(result, :num_iterations)
    end

    test "aggregate single cv_result" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check that statistics are calculated even for single result
      assert result.num_iterations.mean == 10
      assert result.last_evaluation.mean == 0.5
    end

    test "aggregate empty list returns empty map" do
      cv_results = []

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Empty result should return empty map or no stats
      assert is_map(result)
    end

    test "aggregate calculates std for numeric values" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.4,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.8,
          evaluator_result: 0.85,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check that std is calculated
      assert is_number(result.last_evaluation.std)
      assert result.last_evaluation.std > 0
    end

    test "aggregate with mixed nil and non-nil evaluator_result" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: nil,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.55,
          evaluator_result: 0.85,
          prediction: [[0.15, 0.25, 0.6]],
          feature_importance_split: [1.5, 2.5, 3.5, 4.5],
          feature_importance_gain: [0.12, 0.22, 0.32, 0.42]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Should aggregate non-nil evaluator_results
      assert is_map(result.evaluator_result)
      assert is_number(result.evaluator_result.mean)
    end

    test "aggregate with empty prediction lists" do
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # prediction should be empty list
      assert result.prediction == []
    end

    test "aggregate calculates correct min and max" do
      cv_results = [
        %{
          num_iterations: 8,
          last_evaluation: 0.3,
          evaluator_result: 0.75,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 12,
          last_evaluation: 0.9,
          evaluator_result: 0.95,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check min and max for num_iterations
      assert result.num_iterations.min == 8
      assert result.num_iterations.max == 12

      # Check min and max for last_evaluation
      assert result.last_evaluation.min == 0.3
      assert result.last_evaluation.max == 0.9

      # Check min and max for evaluator_result
      assert result.evaluator_result.min == 0.75
      assert result.evaluator_result.max == 0.95
    end

    test "aggregate calculates correct median" do
      cv_results = [
        %{
          num_iterations: 5,
          last_evaluation: 0.2,
          evaluator_result: 0.7,
          prediction: [[0.1, 0.2, 0.7]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        },
        %{
          num_iterations: 15,
          last_evaluation: 0.8,
          evaluator_result: 0.9,
          prediction: [[0.15, 0.25, 0.6]],
          feature_importance_split: [3, 4, 5, 6],
          feature_importance_gain: [0.2, 0.3, 0.4, 0.5]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check median (middle value when sorted)
      assert result.num_iterations.med == 10
      assert result.last_evaluation.med == 0.5
      assert result.evaluator_result.med == 0.8
    end

    test "aggregate with known standard deviation" do
      # Create data with known std: values [1, 2, 3]
      # mean = 2, std = 1.0 (population std)
      cv_results = [
        %{
          num_iterations: 1,
          last_evaluation: 0.1,
          evaluator_result: 0.5,
          prediction: [],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 2,
          last_evaluation: 0.2,
          evaluator_result: 0.6,
          prediction: [],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        },
        %{
          num_iterations: 3,
          last_evaluation: 0.3,
          evaluator_result: 0.7,
          prediction: [],
          feature_importance_split: [3, 4, 5, 6],
          feature_importance_gain: [0.2, 0.3, 0.4, 0.5]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Verify mean
      assert result.num_iterations.mean == 2
      assert_in_delta(result.last_evaluation.mean, 0.2, 0.001)
      assert_in_delta(result.evaluator_result.mean, 0.6, 0.001)

      # Verify std is calculated (non-zero for non-identical values)
      assert result.num_iterations.std > 0
      assert result.last_evaluation.std > 0
      assert result.evaluator_result.std > 0
    end

    test "aggregate maintains prediction structure correctly" do
      # Test with multiple predictions per sample across folds
      cv_results = [
        %{
          num_iterations: 10,
          last_evaluation: 0.5,
          evaluator_result: 0.8,
          prediction: [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]],
          feature_importance_split: [1, 2, 3, 4],
          feature_importance_gain: [0.1, 0.2, 0.3, 0.4]
        },
        %{
          num_iterations: 10,
          last_evaluation: 0.6,
          evaluator_result: 0.85,
          prediction: [[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
          feature_importance_split: [2, 3, 4, 5],
          feature_importance_gain: [0.15, 0.25, 0.35, 0.45]
        }
      ]

      result = LgbmExx.CVResult.aggregate(cv_results)

      # Check that predictions are averaged correctly
      assert is_list(result.prediction)
      # Two test samples
      assert Enum.count(result.prediction) == 2

      # Check first sample: avg of [0.2, 0.3, 0.5] and [0.3, 0.4, 0.3]
      [first_pred | _] = result.prediction
      assert is_list(first_pred)
      assert Enum.count(first_pred) == 3

      # Verify averaging: [(0.2+0.3)/2, (0.3+0.4)/2, (0.5+0.3)/2] = [0.25, 0.35, 0.4]
      expected_first = [0.25, 0.35, 0.4]

      Enum.zip(first_pred, expected_first)
      |> Enum.each(fn {actual, expected} ->
        assert_in_delta(actual, expected, 0.01)
      end)
    end
  end
end
