from explainer import Explainer

if __name__ == "__main__":
    explainer = Explainer(
        explainers=["utopia-payoff", "s_utopia-payoff", "banzhaf", "s_banzhaf", "tau"],
        games=["gwa"],
        configs=[
            ("s_utopia-payoff", "utopia-payoff", "gwa"),
            ("s_banzhaf", "banzhaf", "gwa"),
            ("utopia-payoff", "banzhaf", "gwa"),
            ("utopia-payoff", "tau", "gwa"),
            ("banzhaf", "tau", "gwa"),
        ],
        use_cache=False,
        normalize=True,
        scale_factor=2.0 
    )

    explainer.run()