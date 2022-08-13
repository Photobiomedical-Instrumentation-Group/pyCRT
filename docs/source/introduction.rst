====================
Introduction to pCRT
====================

**pCRT** (alternatively called **rCRT**) is an algorithmically calculated
metric of peripheral perfusion that is analogous to **CRT** (*Capillary Refill
Time*). The acronyms stand for *polarized CRT* or *reflective CRT*, in
reference to the use of polarizers and diffusely reflected light in the
master's thesis [1]_ which this library is based upon.

CRT is defined as the time it takes for a region of skin (typically a finger
or the forearm) to return to its original color after enough pressure is
applied to cause bleaching [2]_. CRT is usually measured by the
examiner's own visual assessment and a chronometer, while pCRT is measured
through the use of a digital camera and digital video processing, resulting in
increased reproducibility and allowing for measurements on all skin phototypes
[1]_.

This library (``pyCRT``) initially sought merely to reimplement the MATLAB
routines developed by Pantojo & Cunha for the aforementioned master's thesis,
but over time its features grew beyond the original work's scope.


.. _algorithm:

Algorithm for calculation
=========================



PyrCRT's data model
===================


References
==========

.. [1] de Souza, Raquel Pantojo, and George Cunha Cardoso. "Skin color independent robust assessment of capillary refill time." arXiv preprint arXiv:2102.13611 (2021).

.. [2] Pickard, Amelia, Walter Karlen, and J. Mark Ansermino. "Capillary refill time: is it still a useful clinical sign?." Anesthesia & Analgesia 113.1 (2011): 120-123.
