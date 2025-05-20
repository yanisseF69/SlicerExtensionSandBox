import os
import slicer

class PythonDependencyChecker(object):
  """
  Class responsible for installing the Modules dependencies
  """

  @classmethod
  def areDependenciesSatisfied(cls):
    try:
      import langchain_huggingface
      import llama_cpp
      import faiss
      import fastapi
      import uvicorn

      return True

    except ImportError:
      return False

  @classmethod
  def installDependenciesIfNeeded(cls, progressDialog=None):
    if cls.areDependenciesSatisfied():
      return

    progressDialog = progressDialog or slicer.util.createProgressDialog(maximum=0)
    progressDialog.labelText = "Installing PyTorch"

    os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
    os.environ["FORCE_CMAKE"] = "1"

    for dep in ["llama-cpp-python", "fastapi", "uvicorn", "langchain_huggingface", "langchain_community", "hf-xet", "faiss-cpu"]:
      progressDialog.labelText = "Installing " + dep
      slicer.util.pip_install(dep)