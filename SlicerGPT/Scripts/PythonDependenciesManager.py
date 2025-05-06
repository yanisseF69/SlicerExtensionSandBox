import os
import slicer

class PythonDependencyChecker(object):
  """
  Class responsible for installing the Modules dependencies
  """

  @classmethod
  def areDependenciesSatisfied(cls):
    try:
      from packaging import version
      import transformers
      import torch
      import langchain_community
      import langchain_huggingface
      import faiss


      # Make sure Transformers version is compatible with the model who will be used
      return version.parse(transformers.__version__) >= version.parse("4.51.0")
    except ImportError:
      return False

  @classmethod
  def installDependenciesIfNeeded(cls, progressDialog=None):
    if cls.areDependenciesSatisfied():
      return

    progressDialog = progressDialog or slicer.util.createProgressDialog(maximum=0)
    progressDialog.labelText = "Installing PyTorch"

    try:
      # Try to install the best available pytorch version for the environment using the PyTorch Slicer extension
      import PyTorchUtils
      PyTorchUtils.PyTorchUtilsLogic().installTorch()
    except ImportError:
      # Fallback on default torch available on PIP
      slicer.util.pip_install("torch")

    for dep in ["langchain_huggingface", "langchain_community", "hf-xet", "faiss-cpu", "transformers>=4.51.0"]:
      progressDialog.labelText = "Installing " + dep
      slicer.util.pip_install(dep)