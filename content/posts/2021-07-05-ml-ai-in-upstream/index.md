---
title: Machine Learning & AI for Predictive Exploration
author: Jayanth Boddu
date: 2021-07-05
hero: ./yana_karnataka.png
excerpt: AI trends and adoption themes in Upstream and resources to get started with AI in Geoscience.
---


Photo : Yana Rocks,Karnataka, Dec 23, 2018

Artificial Intelligence (AI) is the ability of machines to perform tasks that conventionally required human cognition.

AI is estimated to create economic value of $ 13 trillion by 2030 with an impact equivalent to $ 173 billion to oil and gas industry (Ref : [AI for everyone , Andrew Ng](https://www.coursera.org/learn/ai-for-everyone)). This huge disruption potential of AI is driving current business strategies to consider digital use cases in ‘Predictive Exploration’ . AI is the core component of Predictive Exploration.

> AI is the core component of Predictive Exploration.

This article provides a brief summary of AI, current AI opportunities and trends in E&P industry, upstream AI adoption themes. It provides resources to get started on understanding and implementing AI solutions for geoscience.

Artificial intelligence is a sub field of data science , the science of extracting knowledge & insights from data. Machine learning (ML) , an AI paradigm is attributed to most of the recent progress in the field of AI. ML teaches machines to perform specific decision tasks without explicit programming. ML techniques can be broadly classified into supervised ML and unsupervised ML. Supervised ML can be succinctly summarised as — teaching a machine to learn a mapping from Input to output ( A to B ) by training it with several examples of (A,B). Once the training is complete , the machine is said to have learned a ‘model’. This model could then be used to predict B, given A.

Unsupervised learning is useful for initial data exploration to find interesting structures in data.

> AI is a general purpose technology for automating tasks in a workflow.

ML predictions ( if reasonably accurate ), could be used for automation of tasks in existing human centric workflows. Hence, AI could be understood as a general purpose technology for automating tasks in a workflow. The deliverable of an AI implementation and Supervised ML in particular is a task specific model, that would automatically predict an output for a given input. However, it is necessary to emphasise that current AI solutions are very task specific. Models trained on one task are suited to performing the said task only.

> Current AI solutions are very task specific.

In its current state, AI can automate any task / decision which takes a human about 1–2 minutes to make. But there is also a negative test of appropriateness of applying AI to a particular task — Does the task consist of well defined rules and / or empirical relationships ? If yes, AI might not bring value to the task. AI is useful for tasks with a lot data ( examples of input — output relationship ) and complex relationships between input and output. The value that AI brings to these tasks is learning patterns and relationships present in existing examples and producing statistically consistent outputs given new inputs. Mathematically speaking, ML models are universal function approximators that learn complex functions from observations. This makes ML models better at learning from ‘unknown knowns’ and ‘known knowns’ present in data and predicting outputs better and faster than humans.

> AI makes prediction cheaper.

From an economic standpoint, AI makes prediction ( a fundamental component of intelligence ) cheaper. A lowered cost of prediction has made businesses re-assess their current workflows to identify tasks that were previously processed with default assumptions. AI gives businesses more choices. Also, since prediction complements judgement which drives business decisions, AI has been transforming businesses by automating repetitive tasks , accelerating workflows and making the businesses more data driven. ( Ref : [Prediction Machines](https://www.goodreads.com/book/show/36484703-prediction-machines) )

> The core goal of Oil & Gas 4.0 is to achieve greater business value through adoption of digital technologies.

The ongoing evolution of the E&P industry, dubbed as Oil & Gas 4.0 has one core goal — to achieve greater business value through adoption of advanced digital technologies. Among these, AI & ML have shown huge potential in improving efficiency by accelerating processes and reducing risk.

> AI & ML can help accelerate and de-risk processes in upstream, which is the most capital intensive and uncertainty ridden part of E&P.

In particular, the upstream sector of E&P is capital intensive and characterised by high risk and uncertainty. Further to exacerbate the challenges, upstream processes rely on expert knowledge , involve subjective perception and experience based decision making, are time and resource intensive , sometimes have no objective measurement criteria.

Hence, there is huge potential for ML / AI tools to not only accelerate and de-risk processes but also establish quality baselines and bring consistency to decision making inputs in upstream.

> The E&P industry is using AI to reduce Time to Value & build Living Earth Models.

E&P industry is already seeing a few broad themes in exploiting AI tools namely — reducing the ‘Time to Value’ of the gathered data and building a ‘Living Earth Model’ which automatically updates with new knowledge. (Ref : [Ep 86 : Applying ML and AI to geosciences, Seismic Soundoff](https://soundcloud.com/seismicsoundoff/86-applying-machine-learning-and-ai-to-the-geosciences?utm_source=clipboard&utm_campaign=wtshare&utm_medium=widget&utm_content=https%253A%252F%252Fsoundcloud.com%252Fseismicsoundoff%252F86-applying-machine-learning-and-ai-to-the-geosciences) ). Further, most use cases have been targeted towards

*   Data driven studies as a faster and less accurate alternative physics driven studies.
*   Providing baseline interpretation metrics for subjective results.
*   Automating data quality checks.

In line with the industry needs, the most successful use cases of ML / AI in upstream have been the ones which provide tangible process acceleration benefits and significant reduction of human errors in mapping hydrocarbon targets.

*   Tools for automated mapping of reservoir rock properties over an oil region ( accelerated from several weeks to several seconds )
*   Tools for extracting geological information from well logs ( 100 + times speed up )
*   Tools for rock typing based on images of rock samples extracted from wells ( ~1,000,000 times speed up )

(Ref : [Artificial Intelligence in Oil & Gas Upstream : Trends , Challenges and scenarios for the future](https://doi.org/10.1016/j.egyai.2020.100041) )

As a compliment to industry adoption, the last three years has seen increased activity in the scientific community in applying ML to geoscience. Recent times have also seen a significant AI drive by professional bodies like SPE, EAGE, SPWLA, SEG. The number of quality research publications in scientific journals and conference proceedings has seen a steady rise. (Look at [EarthArxiv](https://eartharxiv.org) ).

The industry’s estimate of AI’s current value can be judged by high reward ( $ 100,000 ) competitions like ‘[Salt Segmentation in Seismic Cubes’](https://www.kaggle.com/c/tgs-salt-identification-challenge). Some other noteworthy competitions in similar vein are

*   [Lithology Prediction Challenge by SEG](https://github.com/seg/2016-ml-contest)
*   [Synthetic Sonic Log Generation by SPWLA](https://github.com/pddasig/Machine-Learning-Competition-2020)
*   [Seismic Facies Identification by SEAM / SEG](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge#submission)
*   [Force 2020 : Lithology Prediction Competition](https://terranubis.com/datainfo/FORCE-ML-Competition-2020-Synthetic-Models-and-Wells)

The code implementations of most competitions have been open sourced to tinker and build on.

AI for geoscience is further augmented by community contributions to developing ML applications with business impact (Ex : [SoftwareUnderground](https://softwareunderground.org) , [AgileScientific](https://agilescientific.com) ) and tutorials for new entrants ( Ex : [SEG Tutorials](https://github.com/seg/tutorials), [Subsurface Machine Learning C](https://www.youtube.com/watch?v=5kBS5ThMHcU&list=PLG19vXLQHvSC2ZKFIkgVpI9fCjkN38kwf)ourse by GeostatsGuy ).

Enormous interpreted datasets have also been released under creative commons license .

*   Datasets on [DataUnderground](https://agilescientific.com)
*   [Volve](https://www.equinor.com/en/what-we-do/digitalisation-in-our-dna/volve-field-data-village-download.html) Digital Village by Equinor
*   [Opendetect](https://terranubis.com)
*   [F3 Interpretation Dataset](https://zenodo.org/record/1471548#.YM2hvS0RpQI)
*   [Facies Mark](https://ieee-dataport.org/open-access/facies-mark-machine-learning-benchmark-facies-classification)
*   [StData-12](https://doi.org/10.1093/gji/ggz226)

I hope this article helps the community effort to accelerate Machine Learning applications in geoscience.
