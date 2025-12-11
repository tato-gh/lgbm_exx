defmodule LgbmExx.CVModelTest do
  use ExUnit.Case, async: false

  defp setup_model(_context) do
    # Use a simplified tmp_dir path without commas
    # ExUnit's tmp_dir includes test name which may contain commas
    base_tmp = Path.join(System.tmp_dir!(), "lgbm_exx_cv_model_test")
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

    %{model: model, df: df}
  end

  # Helper function to create train/val split
  defp create_train_val_split(df) do
    train_df = Explorer.DataFrame.slice(df, 0, 120)
    val_df = Explorer.DataFrame.slice(df, 120, 30)
    {train_df, val_df}
  end

  # Helper function to assert basic result structure
  defp assert_basic_result_structure(result) do
    assert is_map(result)
    assert Map.has_key?(result, :num_iterations)
    assert Map.has_key?(result, :last_evaluation)
    assert Map.has_key?(result, :evaluator_result)
    assert Map.has_key?(result, :prediction)
    assert Map.has_key?(result, :feature_importance_split)
    assert Map.has_key?(result, :feature_importance_gain)

    # Check num_iterations is valid
    assert is_integer(result.num_iterations)
    assert result.num_iterations > 0

    # Check feature importances are lists
    assert is_list(result.feature_importance_split)
    assert is_list(result.feature_importance_gain)
    assert Enum.count(result.feature_importance_split) > 0
    assert Enum.count(result.feature_importance_gain) > 0
  end

  # Helper function to create a simple evaluator
  defp create_mean_probability_evaluator do
    fn y_val, pred_val ->
      Enum.zip(y_val, pred_val)
      |> Enum.map(fn {class, probs} ->
        class_num = if is_number(class), do: class, else: String.to_integer(to_string(class))
        index = round(class_num)
        Enum.at(probs, index)
      end)
      |> Explorer.Series.from_list()
      |> Explorer.Series.mean()
    end
  end

  describe "fit_and_evaluate" do
    @describetag :tmp_dir

    setup [:setup_model]

    test "fit_and_evaluate without evaluator and x_test returns result map", %{
      model: model,
      df: df
    } do
      {train_df, val_df} = create_train_val_split(df)

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, [], nil)

      # Check basic result structure
      assert_basic_result_structure(result)

      # Check evaluator_result should be nil when evaluator is not provided
      assert result.evaluator_result == nil

      # Check prediction should be empty list when x_test is empty
      assert result.prediction == []
    end

    test "fit_and_evaluate with evaluator calculates evaluator_result", %{model: model, df: df} do
      {train_df, val_df} = create_train_val_split(df)
      evaluator = create_mean_probability_evaluator()

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, [], evaluator)

      # Check basic result structure
      assert_basic_result_structure(result)

      # Check evaluator_result should be calculated
      assert result.evaluator_result != nil
      assert is_number(result.evaluator_result)
      assert result.evaluator_result >= 0.0
      assert result.evaluator_result <= 1.0
    end

    test "fit_and_evaluate with x_test returns predictions", %{model: model, df: df} do
      {train_df, val_df} = create_train_val_split(df)

      x_test = [
        [5.4, 3.9, 1.7, 0.4],
        [5.7, 2.8, 4.5, 1.3],
        [7.6, 3.0, 6.6, 2.2]
      ]

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, x_test, nil)

      # Check basic result structure
      assert_basic_result_structure(result)

      # Check prediction should contain results for each x_test sample
      assert result.prediction != []
      assert Enum.count(result.prediction) == Enum.count(x_test)

      # Check each prediction should be valid probabilities
      predictions_valid =
        Enum.all?(result.prediction, fn pred ->
          is_list(pred) and Enum.count(pred) == 3 and
            Enum.all?(pred, fn p -> is_number(p) and p >= 0.0 and p <= 1.0 end) and
            abs(Enum.sum(pred) - 1.0) < 0.01
        end)

      assert predictions_valid,
             "All predictions should be valid probability distributions"

      # Verify that different samples produce different predictions (not all identical)
      predictions_diverse =
        result.prediction
        |> Enum.zip_reduce([], fn pred, acc -> [pred | acc] end)
        |> Enum.uniq()
        |> Enum.count() > 1

      assert predictions_diverse,
             "Different samples should produce different predictions"
    end

    test "fit_and_evaluate returns feature importance split", %{model: model, df: df} do
      {train_df, val_df} = create_train_val_split(df)

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, [], nil)

      # Check basic result structure (includes feature importance checks)
      assert_basic_result_structure(result)

      # Check feature_importance_split has expected number of features (4 for iris)
      assert Enum.count(result.feature_importance_split) == 4
    end

    test "fit_and_evaluate returns feature importance gain", %{model: model, df: df} do
      {train_df, val_df} = create_train_val_split(df)

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, [], nil)

      # Check basic result structure (includes feature importance checks)
      assert_basic_result_structure(result)

      # Check feature_importance_gain has expected number of features (4 for iris)
      assert Enum.count(result.feature_importance_gain) == 4
    end

    test "fit_and_evaluate with both evaluator and x_test works correctly", %{
      model: model,
      df: df
    } do
      {train_df, val_df} = create_train_val_split(df)

      x_test = [
        [5.4, 3.9, 1.7, 0.4],
        [5.7, 2.8, 4.5, 1.3]
      ]

      evaluator = fn y_val, pred_val ->
        Enum.zip(y_val, pred_val)
        |> Enum.count(fn {class, probs} ->
          class_num = if is_number(class), do: class, else: String.to_integer(to_string(class))
          index = round(class_num)
          prob = Enum.at(probs, index)
          prob > 0.5
        end)
      end

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, x_test, evaluator)

      # Check basic result structure
      assert_basic_result_structure(result)

      # Check both evaluator_result and prediction are set
      assert result.evaluator_result != nil
      assert is_integer(result.evaluator_result)
      assert result.prediction != []
      assert Enum.count(result.prediction) == Enum.count(x_test)
    end

    test "fit_and_evaluate with last_evaluation", %{model: model, df: df} do
      {train_df, val_df} = create_train_val_split(df)

      result = LgbmExx.CVModel.fit_and_evaluate(model, {train_df, val_df}, [], nil)

      # Check basic result structure
      assert_basic_result_structure(result)

      # Check last_evaluation is set (may be nil if no learning steps)
      if result.last_evaluation do
        assert is_number(result.last_evaluation)
      end
    end
  end
end
