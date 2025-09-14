import argparse
from rich import print

def main():
    p = argparse.ArgumentParser(description="NeuroVector Sim CLI")
    p.add_argument("--model", default="snn_mnist")
    p.add_argument("--device", default="rram_simple")
    p.add_argument("--noise", nargs="*", default=[])
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--report", default="reports/report.json")
    args = p.parse_args()

    print("[bold]NeuroVector Sim[/bold]")
    print(f" model: {args.model}")
    print(f" device: {args.device}")
    print(f" noise: {args.noise}")
    print(f" dataset: {args.dataset}")
    print(f" epochs: {args.epochs}")
    print(f" report: {args.report}")
    print("smoke test OK (wiring pending)")

if __name__ == "__main__":
    main()
