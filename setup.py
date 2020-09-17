"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2. Unpin specific versions from setup.py (like isort).

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

8. Add the release version to docs/source/_static/js/custom.js and .circleci/deploy.sh

9. Update README.md to redirect to correct documentation.
"""

import shutil
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale transformers.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "transformers.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "If you recently updated transformers to 3.0 or later, this is expected,\n"
            "but it may prevent transformers from installing in editable mode.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now.\n\n"
            "See https://github.com/pypa/pip/issues/5466 for details.\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)


extras = {}

extras["ja"] = ["fugashi>=1.0", "ipadic>=1.0.0,<2.0", "unidic_lite>=1.0.7", "unidic>=1.0.2"]
extras["sklearn"] = ["scikit-learn"]

# keras2onnx and onnxconverter-common version is specific through a commit until 1.7.0 lands on pypi
extras["tf"] = [
    "tensorflow",
    "onnxconverter-common",
    "keras2onnx"
    # "onnxconverter-common @ git+git://github.com/microsoft/onnxconverter-common.git@f64ca15989b6dc95a1f3507ff6e4c395ba12dff5#egg=onnxconverter-common",
    # "keras2onnx @ git+git://github.com/onnx/keras-onnx.git@cbdc75cb950b16db7f0a67be96a278f8d2953b48#egg=keras2onnx",
]
extras["tf-cpu"] = [
    "tensorflow-cpu",
    "onnxconverter-common",
    "keras2onnx"
    # "onnxconverter-common @ git+git://github.com/microsoft/onnxconverter-common.git@f64ca15989b6dc95a1f3507ff6e4c395ba12dff5#egg=onnxconverter-common",
    # "keras2onnx @ git+git://github.com/onnx/keras-onnx.git@cbdc75cb950b16db7f0a67be96a278f8d2953b48#egg=keras2onnx",
]
extras["torch"] = ["torch"]
extras["onnxruntime"] = ["onnxruntime>=1.4.0", "onnxruntime-tools>=1.4.2"]

extras["serving"] = ["pydantic", "uvicorn", "fastapi", "starlette"]
extras["all"] = extras["serving"] + ["tensorflow", "torch"]

extras["testing"] = ["pytest", "pytest-xdist", "timeout-decorator", "psutil", "parameterized", "faiss-cpu", "datasets"]
# sphinx-rtd-theme==0.5.0 introduced big changes in the style.
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme==0.4.3", "sphinx-copybutton"]
extras["quality"] = ["black >= 20.8b1", "isort >= 5", "flake8 >= 3.8.3"]
extras["dev"] = extras["testing"] + extras["quality"] + extras["ja"] + ["scikit-learn", "tensorflow", "torch"]

setup(
    name="transformers",
    version="3.1.0",
    author="Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Sam Shleifer, Patrick von Platen, Google AI Language Team Authors, Open AI team Authors, Facebook AI Authors, Carnegie Mellon University Authors",
    author_email="thomas@huggingface.co",
    description="State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch tensorflow BERT GPT GPT-2 google openai CMU",
    license="Apache",
    url="https://github.com/huggingface/transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        "tokenizers == 0.8.1.rc2",
        # dataclasses for Python versions that don't have it
        "dataclasses;python_version<'3.7'",
        # utilities from PyPA to e.g. compare versions
        "packaging",
        # filesystem locks e.g. to prevent parallel downloads
        "filelock",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm >= 4.27",
        # for OpenAI GPT
        "regex != 2019.12.17",
        # for XLNet
        "sentencepiece != 0.1.92",
        # for XLM
        "sacremoses",
    ],
    extras_require=extras,
    entry_points={
        "console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]
    },
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
