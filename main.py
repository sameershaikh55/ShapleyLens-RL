from explainer import Explainer

if __name__ == "__main__":
    explainer = Explainer(
        explainers=["utopia-payoff", "banzhaf", "gately"],
        games=["battleship"],
        configs=[
            ("gately", "banzhaf", "battleship"),
            ("utopia-payoff", "banzhaf", "battleship"),
            ("utopia-payoff", "gately", "battleship"),
        ],
        use_cache=False,
        normalize=True,
    )

    explainer.run()