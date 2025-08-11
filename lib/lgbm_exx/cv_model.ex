defmodule LgbmExx.CVModel do
  @moduledoc """
  """

  @model_name "cv"

  def fit_and_evaluate(model, {train_df, val_df}, x_test, evaluator) do
    %{y_name: y_name, x_names: x_names} = Map.new(model.parameters)

    model_cv = LgbmEx.fit(@model_name, {train_df, val_df}, {y_name, x_names}, model.parameters)

    evaluator_result =
      if evaluator do
        x_val = val_df[x_names]
        y_val = val_df[y_name] |> Explorer.Series.to_list()
        pred_val = LgbmEx.predict(model_cv, x_val)
        evaluator.(y_val, pred_val)
      end

    last_evaluation =
      if model_cv.learning_steps != [] do
        List.last(model_cv.learning_steps) |> elem(1)
      else
        nil
      end

    %{
      num_iterations: model_cv.num_iterations,
      last_evaluation: last_evaluation,
      evaluator_result: evaluator_result,
      prediction: LgbmEx.predict(model_cv, x_test),
      feature_importance_split: model_cv.feature_importance_split,
      feature_importance_gain: model_cv.feature_importance_gain
    }
  end
end
