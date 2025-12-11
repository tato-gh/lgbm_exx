defmodule LgbmExx.MixProject do
  use Mix.Project

  @version "0.0.1"

  def project do
    [
      app: :lgbm_exx,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:lgbm_ex, "~> 0.0.2", github: "tato-gh/lgbm_ex"},
      # Data
      {:explorer, "~> 0.8.1"},
      {:scholar, "~> 0.2.1"},
      {:nx, "~> 0.10.0"},
      # Test
      {:mix_test_observer, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end
end
