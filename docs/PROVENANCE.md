# Provenance and attribution

LPWitness originated in Kirill's research/software project. The prototype was
implemented and tested with Níl, an OpenAI AI assistant, and subsequently
reviewed internally by separate AI-assisted checks. It was not solely unaided
human implementation. Kirill Shevtsov maintains the public project; see the repository’s citation metadata. No external institution or project is
an author, sponsor, or endorser by virtue of being named here.

Standard rational linear algebra, LP weak duality and Farkas certificates are
prior art. Related tools include [SoPlex](https://soplex.zib.de/),
[SCIP exact mode](https://scipopt.org/doc/html/EXACT.php),
[VIPR](https://github.com/scipopt/vipr), and
[MONGOOSE/ExternalChecker](https://cb.csail.mit.edu/mongoose/). LPWitness is a
small reproducible verifier for deliberately limited continuous LP witnesses,
not a claim to have invented an exact proof system.

The example based on SciPy #20336 redistributes coefficient data extracted
from one BSD-3-Clause regression; see the exact model contracts,
[pinned source notice](../examples/original_model_gate/provenance/SOURCE.json)
and retained [SciPy license](../examples/original_model_gate/provenance/SCIPY_LICENSE.txt).
The optional adapter imports SciPy at runtime; it does not vendor SciPy code.
Other bundled examples are author-generated small models or outputs for those
models, not a copy of third-party datasets. The larger private comparison
corpus, original-model source regression text, and vendor executables are not
included in this source tree.

LPWitness’s own code is released under the MIT license. This does not
relicense the SciPy fixture.
