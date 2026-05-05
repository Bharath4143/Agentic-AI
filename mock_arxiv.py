"""
Mock arXiv cs.CL corpus — realistic paper metadata and abstracts
covering major NLP/ML topics for multi-hop reasoning tests.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    categories: List[str]
    citations: List[str] = field(default_factory=list)  # paper_ids this paper cites


PAPERS: List[Paper] = [
    Paper(
        paper_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        authors=["Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan"],
        abstract=(
            "We demonstrate that scaling language models greatly improves task-agnostic, "
            "few-shot performance. GPT-3, an autoregressive language model with 175 billion "
            "parameters, achieves strong performance on many NLP datasets, including translation, "
            "question-answering, and cloze tasks, as well as several tasks that require on-the-fly "
            "reasoning or domain adaptation. Few-shot learning with GPT-3 achieves results "
            "competitive with prior state-of-the-art fine-tuning approaches on certain benchmarks, "
            "while being task-agnostic and requiring no gradient updates or fine-tuning. "
            "We also identify potential harms and limitations of LLMs."
        ),
        year=2020,
        categories=["cs.CL", "cs.LG"],
        citations=["1706.03762", "1810.04805"],
    ),
    Paper(
        paper_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        abstract=(
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. The best "
            "performing models also connect the encoder and decoder through an attention mechanism. "
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "Experiments on two machine translation tasks show these models to be superior in "
            "quality while being more parallelizable and requiring significantly less time to train. "
            "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."
        ),
        year=2017,
        categories=["cs.CL", "cs.LG"],
        citations=[],
    ),
    Paper(
        paper_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        abstract=(
            "We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent language "
            "representation models, BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text by jointly conditioning on both left and right context in all layers. "
            "As a result, the pre-trained BERT model can be fine-tuned with just one additional output "
            "layer to create state-of-the-art models for a wide range of tasks, such as question "
            "answering and language inference, without substantial task-specific architecture modifications. "
            "BERT obtains new state-of-the-art results on eleven natural language processing tasks."
        ),
        year=2018,
        categories=["cs.CL"],
        citations=["1706.03762"],
    ),
    Paper(
        paper_id="2208.11970",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        abstract=(
            "Large pre-trained language models have been shown to store factual knowledge in their "
            "parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. "
            "However, their ability to access and precisely manipulate knowledge is still limited, and "
            "hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. "
            "We explore a general-purpose fine-tuning recipe for RAG models — models that combine "
            "pre-trained parametric and non-parametric memory for language generation. We endow the models "
            "with access to a dense vector index of Wikipedia, queried with a neural retriever. "
            "RAG models achieve state-of-the-art results on open-domain QA, outperforming parametric-only "
            "seq2seq models and task-specific retrieve-and-extract architectures."
        ),
        year=2020,
        categories=["cs.CL", "cs.IR"],
        citations=["1810.04805", "2004.04906"],
    ),
    Paper(
        paper_id="2004.04906",
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        authors=["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis"],
        abstract=(
            "Open-domain question answering relies on efficient passage retrieval to select candidate "
            "contexts, where traditional sparse vector space model (e.g., TF-IDF or BM25) is the "
            "de facto method. In this work, we show that retrieval can be practically implemented "
            "using dense representations alone, where embeddings are learned from a small number of "
            "questions and passages by a simple dual-encoder framework. DPR substantially outperforms "
            "BM25 on multiple open-domain QA datasets. Our results suggest that dense retrieval "
            "can replace the sparse retrieval component in existing open-domain QA systems, "
            "achieving state-of-the-art results on Natural Questions, TriviaQA, and WebQuestions."
        ),
        year=2020,
        categories=["cs.CL", "cs.IR"],
        citations=["1810.04805"],
    ),
    Paper(
        paper_id="2210.11610",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        authors=["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran"],
        abstract=(
            "While large language models (LLMs) have demonstrated impressive capabilities across "
            "tasks in language understanding and interactive decision making, their abilities for "
            "reasoning and acting have largely been studied as separate topics. In this paper, we "
            "explore the use of LLMs to generate both reasoning traces and task-specific actions "
            "in an interleaved manner, allowing for greater synergy between the two. ReAct prompts "
            "LLMs to generate verbal reasoning traces and actions pertaining to a task in an "
            "interleaved fashion. ReAct overcomes issues of hallucination and error propagation prevalent "
            "in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates "
            "interpretable task-solving trajectories that humans can inspect and control."
        ),
        year=2022,
        categories=["cs.CL", "cs.AI"],
        citations=["2005.14165", "2201.11903"],
    ),
    Paper(
        paper_id="2201.11903",
        title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        authors=["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma"],
        abstract=(
            "We explore how generating a chain of thought — a series of intermediate reasoning steps — "
            "significantly improves the ability of large language models to perform complex reasoning. "
            "In particular, we show how such reasoning abilities emerge naturally in sufficiently large "
            "language models via a simple method called chain-of-thought prompting, where a few chain-of-"
            "thought demonstrations are provided as exemplars in prompting. Experiments on three large "
            "language models show that chain-of-thought prompting improves performance on a range of "
            "arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. "
            "For example, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves "
            "state-of-the-art accuracy on the GSM8K benchmark of math word problems."
        ),
        year=2022,
        categories=["cs.CL", "cs.AI"],
        citations=["2005.14165"],
    ),
    Paper(
        paper_id="2302.07842",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        authors=["Tiago Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu"],
        abstract=(
            "Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few "
            "examples or textual instructions, especially at scale. Despite this, they struggle with "
            "basic functionality, such as arithmetic or factual lookup, where much simpler and smaller "
            "specialized models vastly outperform them. In this paper, we show that LMs can teach "
            "themselves to use external tools via simple APIs and that this in a self-supervised way "
            "requiring nothing more than a handful of demonstrations for each API. Toolformer achieves "
            "substantially improved zero-shot performance across a variety of downstream tasks, often "
            "competitive with much larger models, without sacrificing its core language modeling abilities."
        ),
        year=2023,
        categories=["cs.CL", "cs.AI"],
        citations=["2005.14165", "2210.11610"],
    ),
    Paper(
        paper_id="2305.10601",
        title="RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment",
        authors=["Hanze Dong", "Wei Xiong", "Deepanshu Goyal", "Rishi Panchal"],
        abstract=(
            "In this paper, we introduce a new algorithm RAFT (Reward rAnked FineTuning) that aligns "
            "generative models with a reward function. RAFT selects the generated samples with high "
            "reward scores and uses them to finetune the model. The key insight is to exploit the "
            "reward signal in a self-play manner. RAFT is highly applicable to both single-modal "
            "generation tasks (such as text generation) and multi-modal generation tasks (such as "
            "text-to-image generation). We evaluate RAFT on the Helpful and Harmless (HH) dataset "
            "and show RAFT achieves better performance than supervised fine-tuning and RLHF methods."
        ),
        year=2023,
        categories=["cs.CL", "cs.LG"],
        citations=["2005.14165", "1810.04805"],
    ),
    Paper(
        paper_id="2307.09288",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        authors=["Hugo Touvron", "Louis Martin", "Kevin Stone", "Peter Albert"],
        abstract=(
            "We develop and release Llama 2, a collection of pretrained and fine-tuned large language "
            "models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned "
            "LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform "
            "open-source chat models on most benchmarks we tested, and based on our human evaluations "
            "for helpfulness and safety, may be a suitable substitute for closed-source models. "
            "We provide a detailed description of our approach to fine-tuning and safety improvements "
            "of Llama 2-Chat in order to enable the community to build on our work and contribute to "
            "the responsible development of LLMs."
        ),
        year=2023,
        categories=["cs.CL"],
        citations=["2005.14165", "2302.07842"],
    ),
    Paper(
        paper_id="2310.06825",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        authors=["Akari Asai", "Zeqiu Wu", "Yizhong Wang", "Avirup Sil", "Hannaneh Hajishirzi"],
        abstract=(
            "Despite their remarkable capabilities, large language models (LLMs) often produce "
            "responses containing factual inaccuracies due to their reliance on the parametric "
            "knowledge they internalize during training. Retrieval-Augmented Generation (RAG), an "
            "ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such "
            "issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved "
            "passages, regardless of whether retrieval is necessary, or need to retrieve different passages, "
            "diminishes LM versatility or can lead to unhelpful response generation. We introduce Self-RAG, "
            "a framework that enhances an LM's quality and factuality through retrieval and self-reflection. "
            "Self-RAG outperforms vanilla RAG and ChatGPT on multiple open-domain QA tasks."
        ),
        year=2023,
        categories=["cs.CL", "cs.IR"],
        citations=["2208.11970", "2004.04906"],
    ),
    Paper(
        paper_id="2401.10020",
        title="RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture",
        authors=["Angels Balaguer", "Vinamra Benara", "Renato Cunha", "Roberto Estevão"],
        abstract=(
            "There are two common ways in which developers are using Large Language Models (LLMs) to "
            "create question and answer systems that are accurate and relevant to their business: "
            "Retrieval-Augmented Generation (RAG) and Fine-Tuning. RAG augments the prompt with the "
            "external data, while fine-tuning incorporates the additional knowledge into the model itself. "
            "We propose a pipeline for fine-tuning and RAG, and present the tradeoffs of both for multiple "
            "popular LLMs, including Llama2-13B, GPT-3.5, and GPT-4. Our pipeline consists of multiple "
            "stages, including extracting information from PDFs, generating questions and answers, "
            "fine-tuning the model, and evaluating its performance. We measure QA performance, model "
            "perception, and subjective user quality scores."
        ),
        year=2024,
        categories=["cs.CL", "cs.AI"],
        citations=["2208.11970", "2307.09288"],
    ),
    Paper(
        paper_id="2404.16130",
        title="ARAGOG: Advanced RAG Output Grading",
        authors=["Matouš Eibich", "Shivay Nagpal", "Alexander Fred-Ojala"],
        abstract=(
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval "
            "with language model generation to produce more accurate and contextually relevant responses. "
            "Despite its promise, RAG systems often struggle with faithfulness — the degree to which "
            "generated responses accurately reflect retrieved information. We propose ARAGOG, a framework "
            "for evaluating RAG output quality along three dimensions: answer correctness, faithfulness to "
            "source documents, and citation precision. Our evaluation reveals significant variance in RAG "
            "system quality across different retrieval strategies (sparse, dense, hybrid), chunking methods, "
            "and reranking approaches. We release benchmark datasets and evaluation scripts to the community."
        ),
        year=2024,
        categories=["cs.CL", "cs.IR"],
        citations=["2208.11970", "2310.06825"],
    ),
    Paper(
        paper_id="2312.10997",
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        authors=["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kangxiang Jia"],
        abstract=(
            "Large Language Models (LLMs) demonstrate significant capabilities but face challenges "
            "such as hallucination, outdated knowledge, and non-transparent, untraceable reasoning "
            "processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by "
            "incorporating knowledge from external databases. This enhances the accuracy and credibility "
            "of the models, particularly for knowledge-intensive tasks, and allows for continuous "
            "knowledge updates and integration of domain-specific information. RAG synergistically "
            "merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. "
            "This survey categorizes RAG into Naive RAG, Advanced RAG, and Modular RAG paradigms, "
            "reviewing the tripartite foundation of retrieval, generation, and augmentation."
        ),
        year=2023,
        categories=["cs.CL", "cs.IR"],
        citations=["2208.11970", "2004.04906", "2310.06825"],
    ),
    Paper(
        paper_id="2305.06983",
        title="HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace",
        authors=["Yongliang Shen", "Kaitao Song", "Xu Tan", "Dongsheng Li"],
        abstract=(
            "Solving complicated AI tasks with different domains and modalities is a key step toward "
            "artificial general intelligence. While there are numerous AI models available for different "
            "domains and modalities, they cannot handle complicated AI tasks. Considering large language "
            "models (LLMs) have exhibited exceptional ability in language understanding, generation, "
            "interaction, and reasoning, we advocate that LLMs could act as a controller to manage "
            "existing AI models to solve AI tasks and language could be a generic interface to empower "
            "this. Based on this philosophy, we present HuggingGPT, a system that leverages LLMs to "
            "connect various AI models in machine learning communities to solve AI tasks."
        ),
        year=2023,
        categories=["cs.CL", "cs.AI"],
        citations=["2210.11610", "2302.07842"],
    ),
]


def get_all_papers() -> List[Paper]:
    return PAPERS


def get_paper_by_id(paper_id: str) -> Paper | None:
    for p in PAPERS:
        if p.paper_id == paper_id:
            return p
    return None
