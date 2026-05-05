"""
Agentic RAG CLI
===============
Commands
--------
  ask        Ask a research question (full pipeline)
  benchmark  Run the evaluation benchmark suite
  retrieve   Quick retrieval test (hybrid vs keyword)
  demo       Show a preset multi-hop demo question
"""

from __future__ import annotations

import sys
import time

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG system over arXiv cs.CL corpus",
    add_completion=False,
)
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_retriever = None
_pipeline = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from agentic_rag.modules.retriever import HybridRetriever
        with console.status("[bold cyan]Loading Sentence-BERT model & building FAISS index…"):
            _retriever = HybridRetriever(alpha=0.6)
    return _retriever


def _get_pipeline(top_k: int = 5, max_hops: int = 4):
    global _pipeline
    if _pipeline is None:
        from agentic_rag.modules.pipeline import ReActPipeline
        _pipeline = ReActPipeline(_get_retriever(), top_k=top_k, max_hops=max_hops)
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# ask command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="Research question to answer"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Docs retrieved per hop"),
    max_hops: int = typer.Option(4, "--max-hops", help="Maximum retrieval hops"),
    show_steps: bool = typer.Option(True, "--steps/--no-steps", help="Show ReAct trace"),
    show_docs: bool = typer.Option(False, "--docs/--no-docs", help="Show retrieved docs"),
):
    """Ask a research question using the full agentic RAG pipeline."""
    console.print()
    console.print(Panel(f"[bold white]{query}[/]", title="[cyan]Research Query", border_style="cyan"))

    pipeline = _get_pipeline(top_k=top_k, max_hops=max_hops)

    with console.status("[bold green]Running agentic RAG pipeline…", spinner="dots"):
        result = pipeline.run(query)

    # ── Plan ──────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]Query Plan"))
    console.print(f"[dim]Reasoning:[/dim] {result.plan.reasoning}")
    for i, sq in enumerate(result.plan.sub_queries, 1):
        console.print(f"  [yellow]{i}.[/yellow] {sq}")

    # ── ReAct Steps ───────────────────────────────────────────────────────────
    if show_steps:
        console.print(Rule("[bold blue]ReAct Trace"))
        for step in result.steps:
            console.print(
                Panel(
                    f"[bold]Thought:[/bold] {step.thought}\n\n"
                    f"[bold]Action:[/bold] [green]{step.action}[/green]\n\n"
                    f"[bold]Observation:[/bold]\n{step.observation}",
                    title=f"[blue]Hop {step.hop}[/blue] — {step.sub_query[:60]}",
                    border_style="blue",
                    expand=False,
                )
            )

    # ── Retrieved Docs ────────────────────────────────────────────────────────
    if show_docs:
        console.print(Rule("[bold magenta]Retrieved Documents"))
        doc_table = Table(show_header=True, box=box.SIMPLE_HEAVY)
        doc_table.add_column("Rank", style="dim", width=5)
        doc_table.add_column("Paper ID", style="cyan", width=12)
        doc_table.add_column("Title", width=45)
        doc_table.add_column("Year", width=5)
        doc_table.add_column("Hybrid Score", width=13)
        for i, d in enumerate(result.all_docs[:8], 1):
            doc_table.add_row(
                str(i),
                d.paper.paper_id,
                d.paper.title[:44],
                str(d.paper.year),
                f"{d.hybrid_score:.4f}",
            )
        console.print(doc_table)

    # ── Final Answer ──────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Final Answer"))
    console.print(
        Panel(result.final_answer, title="[green]Synthesized Answer", border_style="green")
    )

    # ── Verification ──────────────────────────────────────────────────────────
    if result.verification:
        v = result.verification
        faith_color = "green" if v.faithful else "red"
        console.print(Rule("[bold]Verification"))
        vtable = Table(show_header=False, box=box.SIMPLE)
        vtable.add_column("Metric", style="bold")
        vtable.add_column("Value")
        vtable.add_row("Faithful", f"[{faith_color}]{'✓' if v.faithful else '✗'} {v.faithful}[/]")
        vtable.add_row("Faithfulness Score", f"{v.faithfulness_score:.3f}")
        vtable.add_row("Groundedness Score", f"{v.groundedness_score:.3f}")
        if v.warnings:
            vtable.add_row("Warnings", "\n".join(v.warnings[:3]))
        console.print(vtable)

    # ── Latency ───────────────────────────────────────────────────────────────
    console.print(Rule("[bold]Latency"))
    lat_table = Table(show_header=False, box=box.SIMPLE)
    lat_table.add_column("Stage", style="bold")
    lat_table.add_column("Time")
    lat_table.add_row("Retrieval", f"{result.retrieval_latency_ms:.1f} ms")
    lat_table.add_row("Generation", f"{result.answer_latency_ms:.1f} ms")
    lat_table.add_row("Total", f"[bold]{result.total_latency_ms:.1f} ms[/bold]")
    console.print(lat_table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# benchmark command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    top_k: int = typer.Option(5, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full evaluation benchmark (hybrid vs keyword baseline)."""
    from agentic_rag.eval.evaluator import Evaluator, BENCHMARK_QUERIES

    pipeline = _get_pipeline(top_k=top_k)
    evaluator = Evaluator(_get_retriever(), pipeline)

    console.print()
    console.print(Rule("[bold cyan]Agentic RAG Benchmark"))
    console.print(f"  Queries: {len(BENCHMARK_QUERIES)}  |  top-k: {top_k}\n")

    results_so_far = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating…", total=len(BENCHMARK_QUERIES))

        def cb(i, total, q):
            progress.update(task, description=f"Query {i+1}: {q[:50]}…", completed=i)

        report = evaluator.run_benchmark(progress_callback=cb)
        progress.update(task, completed=len(BENCHMARK_QUERIES))

    # ── Per-query table ───────────────────────────────────────────────────────
    if verbose:
        console.print(Rule("[bold]Per-Query Results"))
        qt = Table(show_header=True, box=box.SIMPLE_HEAVY)
        qt.add_column("Query", width=42)
        qt.add_column("H-Recall@5", width=11)
        qt.add_column("K-Recall@5", width=11)
        qt.add_column("MRR", width=6)
        qt.add_column("ROUGE-L", width=8)
        qt.add_column("Faith", width=7)
        qt.add_column("Latency(ms)", width=12)
        for qr in report.query_results:
            qt.add_row(
                qr.query[:41],
                f"{qr.hybrid_recall_at_5:.2f}",
                f"{qr.keyword_recall_at_5:.2f}",
                f"{qr.hybrid_mrr:.2f}",
                f"{qr.rouge_l_f1:.2f}",
                f"{qr.faithfulness_score:.2f}",
                f"{qr.total_latency_ms:.0f}",
            )
        console.print(qt)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Benchmark Summary"))

    rt = Table(show_header=True, box=box.ROUNDED, title="Retrieval")
    rt.add_column("Metric", style="bold")
    rt.add_column("Hybrid RAG", style="green")
    rt.add_column("Keyword Baseline", style="yellow")
    rt.add_column("Improvement", style="cyan")

    recall_imp = report.recall_improvement_pct
    mrr_imp = (
        (report.mean_hybrid_mrr - report.mean_keyword_mrr) / max(report.mean_keyword_mrr, 1e-9)
    ) * 100

    rt.add_row(
        "Recall@5",
        f"{report.mean_hybrid_recall_at_5:.3f}",
        f"{report.mean_keyword_recall_at_5:.3f}",
        f"[{'green' if recall_imp >= 0 else 'red'}]{recall_imp:+.1f}%[/]",
    )
    rt.add_row(
        "MRR",
        f"{report.mean_hybrid_mrr:.3f}",
        f"{report.mean_keyword_mrr:.3f}",
        f"[{'green' if mrr_imp >= 0 else 'red'}]{mrr_imp:+.1f}%[/]",
    )
    console.print(rt)

    at = Table(show_header=True, box=box.ROUNDED, title="Answer Quality")
    at.add_column("Metric", style="bold")
    at.add_column("Score", style="green")
    at.add_row("ROUGE-L F1", f"{report.mean_rouge_l:.3f}")
    at.add_row("Faithfulness", f"{report.mean_faithfulness:.3f}")
    at.add_row("Groundedness", f"{report.mean_groundedness:.3f}")
    console.print(at)

    lt = Table(show_header=True, box=box.ROUNDED, title="Latency")
    lt.add_column("Stage", style="bold")
    lt.add_column("Mean (ms)", style="cyan")
    lt.add_row("Retrieval", f"{report.mean_retrieval_latency_ms:.1f}")
    lt.add_row("Generation", f"{report.mean_answer_latency_ms:.1f}")
    lt.add_row("Total (mean)", f"{report.mean_total_latency_ms:.1f}")
    lt.add_row("Total (P95)", f"{report.p95_latency_ms:.1f}")
    console.print(lt)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# retrieve command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query to retrieve documents for"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Show keyword baseline too"),
):
    """Run a quick retrieval test, optionally comparing hybrid vs keyword."""
    retriever = _get_retriever()

    hybrid_docs, h_ms = retriever.retrieve(query, top_k=top_k)
    console.print(f"\n[bold cyan]Hybrid Retrieval[/bold cyan] ({h_ms:.1f} ms)\n")

    ht = Table(box=box.SIMPLE_HEAVY, show_header=True)
    ht.add_column("Rank", width=5)
    ht.add_column("Title", width=50)
    ht.add_column("Year", width=5)
    ht.add_column("Dense", width=7)
    ht.add_column("Sparse", width=7)
    ht.add_column("Hybrid", width=8)
    for d in hybrid_docs:
        ht.add_row(
            str(d.rank),
            d.paper.title[:49],
            str(d.paper.year),
            f"{d.dense_score:.3f}",
            f"{d.sparse_score:.3f}",
            f"[bold]{d.hybrid_score:.3f}[/]",
        )
    console.print(ht)

    if compare:
        kw_docs, k_ms = retriever.keyword_retrieve(query, top_k=top_k)
        console.print(f"\n[bold yellow]Keyword (TF-IDF) Baseline[/bold yellow] ({k_ms:.1f} ms)\n")
        kt = Table(box=box.SIMPLE_HEAVY, show_header=True)
        kt.add_column("Rank", width=5)
        kt.add_column("Title", width=50)
        kt.add_column("Year", width=5)
        kt.add_column("TF-IDF Score", width=13)
        for d in kw_docs:
            kt.add_row(str(d.rank), d.paper.title[:49], str(d.paper.year), f"{d.sparse_score:.3f}")
        console.print(kt)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# demo command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def demo():
    """Run a pre-set multi-hop demonstration query."""
    query = (
        "How do dense retrieval methods and chain-of-thought prompting "
        "together improve open-domain question answering with RAG systems?"
    )
    console.print("\n[bold magenta]Running multi-hop demo…[/bold magenta]\n")
    # Invoke ask programmatically
    ctx = typer.Context(app.registered_commands[0])  # type: ignore
    ask(query=query, top_k=5, max_hops=4, show_steps=True, show_docs=True)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
