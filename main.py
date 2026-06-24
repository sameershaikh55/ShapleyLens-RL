from explainer import Explainer

if __name__ == "__main__":
    explainer = Explainer(
        explainers=["utopia-payoff", "banzhaf", "tau"],
        games=["gwa"],
        configs=[
            ("utopia-payoff", "banzhaf", "gwa"),
            ("utopia-payoff", "tau", "gwa"),
            ("banzhaf", "tau", "gwa"),
        ],
        use_cache=False,
        normalize=True,
    )

    explainer.run()