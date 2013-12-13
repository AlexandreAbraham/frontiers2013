Reviewer 1
==========

Is the method described accurately enough to be reproducible?
-------------------------------------------------------------

No
Overall, the article does not provide a clear view of the use of scikit-learn
for non-experienced readers. With general neuroscience audience is a target,
more clear detail about the organisation of python libraries and scikits would
be helpful. Such readers are likely be confused by the references to specific
functions deep in scipy and numpy, who would gain the impression that strong
experience with scipy is required. Nilearn functionality is mentioned a number
of times despite initially being described initially as "not ready for public
consumption". References are made to functionality such as confound removal and
frequency filtering with no reference to which tools perform these. The
filenames and functions used in the code snippets are not always described
clearly. The result section should describe the use of scikits learn for
particular applications, in a way that demonstrates the features of the toolkit
. However, it instead tends to focus on describing specific methods and results
of implementations of simple analyses using scikits, which are of limited
interest. While this may help non-neuroscience readers, this presentation needs
to provide more insight into the tool.

Answer
......

We agree that a more general introduction of the scikit-learn is
interesting and important. However, the focus of this manuscript is on
neuroimaging specific cases. Indeed, scikit-learn is presented in the
reference paper Pedregosa et al. 2012, and has a very complete online
documentation.

Our goal with this manuscript is to provide enough material so that the
non-experienced can run an estimator on his data. The accompagning
scripts are short, documented, and simple enough to be understood by
non-technical people. We believe that these scripts can be used as a
basis for the reader's own study. Numpy and scipy are huge modules that
proposes a lot of features we do not need. We only highlighted from these
packages the features that we found useful in our field.

The code snippets in the code are short to focus on the interesting lines
of each script, performing the task discussed in the text. Technical
details like module importation are treated directly into the
accompagning scripts, that are fully documented and readable by a
non-experienced user.

We reworded the mentions to nilearn: it is under development and is not
suitable for end users. We have removed all references to nilearn
from the scripts.

Are methods of other authors accurately presented and interpreted?
------------------------------------------------------------------

No
PyMVPA is highly relevant and could be described and compared in more detail.

Answer
......

There are indeed existing solutions to process neuroimaging data with
Python. However, PyMVPA is clearly at a higher level: it provides
abstractions to process neuroimaging data while we only describe
low-level code patterns and usages. We explicitely point to PyMVPA for
these usecases and give a reference to the corresponding paper.


Please add any further comments you have regarding this manuscript.
-------------------------------------------------------------------

Note: I didn't report here typos or formulation remarks from the reviewer.

Intro: This should be made more clear - it has a number of vague and unclear
statements, and fails to clearly position SkiKits-Learn with the python scientific
stack.

4.4 Results These examples and text should focus on the functionality of scikits-learn,
not the outcomes of simple comparisons of methods. Figure: no scale provided. The
figure is not particularly informative in the context of a review of the toolbox.

Answer
......

Intro: we detailed the position of scikit-learn in the Python scientific stack
and why we think it has its place into the neuroimaging landscape.

4.4 Results: We focused on the outcome of the simple methods for
neuroimaging people that are not into machine learning. We want to show
the possibilities opened by scikit-learn. We outlined the fact that our
simple models can be replaced by any more sophisticated model embedded in
scikit-learn.





Reviewer 2
==========

Are relevant methods of other authors mentioned?
------------------------------------------------

No
The authors are strictly focused on the Python software ecosystem. While this
is probably acceptable in the scope of this special issue, I still think it
needs to be mentioned that neuroimaging data analysis with this category of
algorithms is supported by other, non-Python toolboxes too (e.g. Pronto, ...).
Moreover, as scikit-learn itself is not tailored to neuroimaging data analysis
per se, it would be only appropriate to refer to other general-purpose ML
toolkits (e.g. weka, ...). The manuscript also needs a closer look regarding
missing references. For example, the concept of an SVM is integral part of the
manuscript, but no reference to Vapnik or a similar publication is made. Instead
a two-sentence description of the technique is provided only. See below for
additional comments.

Answer
......

We add details to the positionning of scikit-learn with regards to
machine learning and neuroimaging frameworks. However, this manuscript is
not a review article, and we cannot compare in detail with every package.
We also added references to
papers that we found useful for the targeted public (we referenced the
book "The Elements of Statistical Learning" by Hastie et al. instead of
Vapnik for example because it has a more general and didactic approach
regarding machine learning).

Other comments: nilearn positioning
-----------------------------------

I had the impression that this manuscript started out to be an introduction into
nilearn, and later on the scope was changed. It starts by mentioning nilearn as
the thing we are really waiting for, then labels it as "not ready for public
consumption", yet keeps referring to it and most example code in the git
repository relies on some functionality from nilearn. I actually cannot run
searchlight without it being installed. I haven't investigated whether CanICA
absolutely requires nilearn, but in the current example script it seems like it
does. In my opinion the authors need to make up their minds. Either acknowledge
nilearn's current status as "necessarily ready for production" or live without
it. Depending on this decision, it may be necessary to discuss the scopes of
scikit-learn vs nilearn.

Answer
......

This manuscript is not meant to be an introduction to nilearn, and we
reworded its positionning. Nilearn is not needed at all to apply
scikit-learn to neuroimaging data, and this usage is specifically what we
whish to discuss.

We also removed all references to nilearn from our scripts. All scripts can now
run without any dependencies outside of Python scientific and, for methods that
require an external dependency (like CanICA or MELODIC), a precomputed result is
provided.

Other comments: potential alternatives
--------------------------------------

My major point of criticism is that the works presented in this manuscript are
not sufficiently contrasted with potential alternatives. A reader can get the
impression that individual pieces of the Python software ecosystem can work
together nicely and extend the scope of each one of them, without the need for
tailored frameworks. However, for someone who is just interested in machine
learning application on neuroimaging data it may be less obvious why this is an
advantage. Section 1.1 seems to be the only place for this aspect, and it
currently communicates to me: "it doesn't matter" -- Python is at least no worse
than Matlab and R.

Answer
......

We expanded on the potential alternatives in the introduction. However,
most of them operate on a higher level than our code snippets (they
propose abstractions). In the conclusion and in the results paragraph of
the different sections we added a discussion on why the broad variety of
methods in scikit-learn is interesting.

The scripts that we provide enable the use of scikit-learn without a
neuroimaging-specific framework. Taking the original script and changing
the model is very simple, and thus opens the door to analysis approaches
that are not present in neuroimaging-specific libraries.

Other comments: advantages of the scikit-learn
----------------------------------------------

Moreover, the following sections indicate that some non-scikit-learn
functionality is required in order to get data in shape for processing. I am
missing information on how that relates to other general purpose machine
learning toolboxes (weka, orange). Once data is in matrix form, what are the
advantages of staying in the Python world with scikit-learn over other
solutions? I think the arguments in favor are obvious, but they haven't been
made in the manuscript.

Answer
......

Once data is in matrix form, and with proper pre-treatments applied, it is in
fact possible to send it to any machine learning framework. Scikit-learn
benefits of the interactivity of Python and provides a lot of efficient
estimators, as you already suggested above. We highlighted this point in the
manuscript.

Other comments: cite previous Haxby studies
-------------------------------------------

Along the same line: The analysis of the Haxby data set has been made over and
over before. The Princeton MVPA toolbox was first, PyMVPA and Pronto followed,
and I am sure there are more. This would have been a great opportunity to
contrast scikit-learn with all these alternatives on a very concrete
implementation/API level. However, none of these publication have at least been
mentioned in this context. 

Answer
......

Indeed, we have added references to these previous studies.

Other comments: cite model-validation references
------------------------------------------------

In the first third of the manuscript an attempt is made to outline the
pre-processing steps of a typical analysis, as well as to introduce the concept
of model validation. However, neither of them is sufficiently described, or
documented with code in the manuscript. That doesn't have to be done, but
critical references to e.g. Pereira et al. or Mur et al. that explain these
concepts in the neuroimaging domain at a more appropriate level are also
completely missing.

Answer
......

Thanks for noticing this. We added the references.


Other comments: hyperparameter tuning
-------------------------------------

On a related note, section 2.2 seems like an explanation of the concept of
hyperparameter tuning in scikit-learn, yet it is not in the section on
concepts.

Answer
......

Hyperparameter tuning is now explained deeper in the concepts and an example is
provided in the Miyawaki example.

Other comments: temporal compression
------------------------------------

I am not sure about the relative importance of spatial resampling for the
analysis presented in this paper. I'd prefer to see temporal compression methods
being mentioned in the section on 'signal cleaning'.

Answer
......

We briefly mention spatial resampling because it is sometimes a
showstopper for people doing neuroimaging, however we have made this
reference very light. Temporal resampling is not mentioned because it
should be used carefully: the method used to compress time series can
have an impact on the model used afterward to analyze data. This is
beyond the scope of this manuscript.

Other comments: Searchlight balls
---------------------------------

The authors refer to the ROI shape of a searchlight as "balls". I believe the
commonly used term is a "sphere" or a "spherical ROI".

Answer
......

A "sphere" is only a surface. The ball refers to the inside of the sphere. Plus,
this is here used in its topological sense (a ball being a
neighborhood in a vectorial space regarding a particular metric).

Other comments: some part of the papers are too technical
---------------------------------------------------------

When reading the manuscript, there were multiple times when I was unsure what
the intended target audience would be. One was the description of the SVM.
Another example is on page 8: "However, in accordance with our prior knowledge,
L1 regularized models, when properly parametrized, outperform an L2 regularized
estimator." This requires a reference.

Answer
......

In the case of the Miyawaki experiment, the hypothesis is that one pixel will be
correlated will only few voxels in the brain. This is why l1 regularization,
that promotes sparsity, is expected to work better.
We removed the statement you quoted given that standard
deviation on scores is too high to state a statistically significant conclusion.

Other comments: Figure 3 unclear
--------------------------------

The caption of figure 3 talks about "the pixel highlighted". However, panel (a),
(c), and (f) have a t-shaped area of four pixels highlighted. This should be
clarified.

Answer
......

There is a difference here between image pixels and brain voxels. One pixel is
highlighted and, in fact, four voxels are highlighted in the brain
representations. This has been detailed in the corresponding caption and the
related text.

Other comments: missing ICA analysis
------------------------------------

Figure 4 shows the results for different ICA implementations. There should be a
statement on why they are all different. Especially Melodic (probably most
widely used in this domain) vs the others. One can't even tell whether it is a
sign difference, as no colorbar is available. Alternatively, de-emphasize the
relevance of this figure. Currently it says: "On fig. 4 we compare a simple
concat ICA as implemented by the code above to more sophisticated multi-subject
methods, ...". But the actual comparison is left to the reader.

Answer
......

We did not insist on the significance of the comparison, because there is
no accepted criterion or figure of merit to compare ICA approaches.

We added a colorbar, flipped the sign of some maps so that they look alike and
made the only statement we could do out of this analysis: sophisticated methods
present less noise than the simplest group ICA strategy.

Other comments: show the advantages of the scikit-learn
-------------------------------------------------------

I understand that there may not be room for this anymore, but, in my opinion,
one of the most valuable aspects of scikit-learn is the breadth of functionality
without the "frameworkiness" that usually comes with it. This enables quick
prototyping of new ideas. I am sure that scikit-learn has more things like the
grid_to_graph function in its repository. A more comprehensive overview of what
functionality is interesting in the neuroimaging data context would be very well
appreciated. 

Answer
......

We thank the reviewer for stressing this point, which we feel is
important to the paper. Following this recommendation, we highlighted the
scikit-learn versatility all along the paper, gave pointers to other
methods in scikit-learn, and reworked the conclusion to make this point
more central.

