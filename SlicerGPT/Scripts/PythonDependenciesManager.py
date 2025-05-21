import os
import slicer

class InstallationError(Exception):
  def __init__(self, message):
    super().__init__(message)
    self.message = message

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

    try:

      progressDialog = progressDialog or slicer.util.createProgressDialog(maximum=0)
      progressDialog.labelText = "Installing PyTorch"

      os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
      os.environ["FORCE_CMAKE"] = "1"

      for dep in ["llama-cpp-python", "fastapi", "uvicorn", "langchain_huggingface", "langchain_community", "hf-xet", "faiss-cpu"]:
        progressDialog.labelText = "Installing " + dep
        slicer.util.pip_install(dep)
    except Exception as e:
      error = f"Installation failed due to {str(e)}.\nIf the installation of llama_cpp failed, please ensure you have a C compiler installed."
      progressDialog.labelText = error
      raise InstallationError(error)