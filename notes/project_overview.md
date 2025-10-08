# Project: Zero-Shot Jet Classification with LLMs

## Overview
This project explores whether large language models (LLMs) can classify particle jets (quark vs. gluon) **without supervised training**, using only zero-shot or few-shot prompting. The work sits at the intersection of **particle physics** and **AI research**, aiming to test if general-purpose models encode useful inductive biases for jet physics tasks.

## Motivation
- **Physics:** Quark–gluon tagging is a long-standing benchmark in collider physics and a critical ingredient for precision measurements and searches.  
- **AI:** LLMs excel at generalization and in-context reasoning; testing them on jet classification probes whether their broad priors extend to scientific domains.  
- **Science+AI Bridge:** Results will show whether text-formatted physics data can be meaningfully parsed by generalist models, guiding future hybrid pipelines.

## Core Questions
1. Can an LLM distinguish quark vs. gluon jets above random chance in a zero-shot setting?  
2. How do **formatting choices** (bullets, YAML, tables) affect performance?  
3. How sensitive are results to **prompting strategies** (plain zero-shot vs. few-shot)?  
4. Do playful **persona instructions** (e.g., “be an experimental physicist”) measurably shift outcomes?  
5. How do LLM results compare to simple baselines like logistic regression?

## Approach
- **Dataset:** Standard quark–gluon jet tagging benchmark.  
- **Methods:** Transform jets into structured text, query LLMs, capture classification probabilities.  
- **Comparisons:** Random baseline, logistic regression baseline, and LLM zero-shot.  
- **Explorations:** Format ablations, prompt variations, persona effects.  
- **Outputs:** AUC and background rejection metrics, visual comparisons, and notes on robustness.

## Broader Scope
- **Scientific AI:** Demonstrates whether LLM priors extend to raw scientific descriptors.  
- **Methodological Insight:** Identifies which text-based strategies (format, prompt, persona) matter most.  
- **Future Directions:** Opens path to hybrid workflows where LLMs assist domain-specific ML, reasoning, or anomaly detection in physics.  
- **Playful Science:** Persona experiments offer a light-hearted but systematic way to probe model behavior.

---
**Intended audience:** AI agents and collaborators who need a high-level understanding of the project scope.  
**Not included here:** Implementation details, scripts, or timeline.
