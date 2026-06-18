from explainer import Explainer

if __name__ == "__main__":
    explainer = Explainer(
        explainers=["utopia-payoff", "banzhaf", "tau"],
        games=["minesweeper"],
        configs=[
            ("utopia-payoff", "banzhaf", "minesweeper"),
            ("utopia-payoff", "tau", "minesweeper"),
            ("banzhaf", "tau", "minesweeper"),
        ],
        use_cache=False,
        normalize=True,
    )

    explainer.run()