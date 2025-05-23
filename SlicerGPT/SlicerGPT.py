import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import qt

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#
# SlicerGPT
#


class SlicerGPT(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = ("SlicerGPT")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Yanisse FERHAOUI - Institut Pascal"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This module integrates an intelligent chatbot designed to assist 3D Slicer users.
You can ask questions in natural language about the software usage, Python scripting, extensions, or advanced features.
<br><br>
The chatbot uses a local knowledge base (RAG) including documentation, forum content, and source code to provide accurate answers.
<br><br>
See more information in the <a href="https://github.com/organization/projectname#SlicerGPT">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This plugin was initially developed during Yanisse FERHAOUI's final-year internship as part of an academic research project.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SlicerGPT1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SlicerGPT",
        sampleName="SlicerGPT1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SlicerGPT1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SlicerGPT1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SlicerGPT1",
    )

    # SlicerGPT2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SlicerGPT",
        sampleName="SlicerGPT2",
        thumbnailFileName=os.path.join(iconsPath, "SlicerGPT2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SlicerGPT2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SlicerGPT2",
    )


#
# SlicerGPTParameterNode
#


@parameterNodeWrapper
class SlicerGPTParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# SlicerGPTWidget
#


class SlicerGPTWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        if not self.areDependenciesSatisfied():
            error_msg = "Slicer PyTorch, langchain and transformers are required by this plugin.\n" \
                        "Please click on the Download button to download and install these dependencies."
            self.layout.addWidget(qt.QLabel(error_msg))
            downloadDependenciesButton = qt.QPushButton("Download dependencies and restart")
            downloadDependenciesButton.connect("clicked(bool)", self.downloadDependenciesAndRestart)
            downloadDependenciesButton.setCheckable(False)
            self.layout.addWidget(downloadDependenciesButton)
            self.layout.addStretch()
            return

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerGPT.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.

        progressDialog = slicer.util.createProgressDialog(maximum=0)
        progressDialog.labelText = "Downloading & launching model (The first time this action can take a while)"
        self.logic = SlicerGPTLogic()
        self.logic.widget = self

        progressDialog.close()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.prompt.textChanged.connect(self.onPromptTextChanged)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.thinkBox.toggled.connect(self.onThinkBoxToggled)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        logging.info("Cleaning up SlicerGPT module")
        
        if hasattr(self, "logic") and hasattr(self.logic, "proc"):
            # Essayer d'abord d'arrêter le serveur en douceur via une requête HTTP
            try:
                import requests
                # Timeout court pour ne pas bloquer la fermeture
                requests.get("http://127.0.0.1:81/shutdown", timeout=1.0)
                logging.info("Sent shutdown request to server")
            except:
                # Ignorer les erreurs, nous allons forcer l'arrêt de toute façon
                pass
            
            # Vérifier si le processus est en cours d'exécution
            if self.logic.proc.state() == qt.QProcess.Running:
                logging.info("Terminating server process")
                
                # Tenter d'abord un arrêt propre
                self.logic.proc.terminate()
                
                # Attendre un peu pour qu'il s'arrête
                if not self.logic.proc.waitForFinished(3000):  # 3 secondes
                    logging.warning("Server did not terminate gracefully, killing process")
                    self.logic.proc.kill()
                    
                    # Attendre à nouveau pour confirmer
                    if not self.logic.proc.waitForFinished(2000):  # 2 secondes supplémentaires
                        logging.error("Failed to kill server process")
                    else:
                        logging.info("Server process killed")
                else:
                    logging.info("Server process terminated gracefully")
                    
                # Nettoyage supplémentaire
                try:
                    pid = self.logic.proc.processId()
                    if pid > 0:
                        import os, signal
                        try:
                            os.kill(pid, signal.SIGTERM)
                            logging.info(f"Sent SIGTERM to process {pid}")
                        except:
                            pass
                except:
                    pass
            
            # S'assurer que tous les pipes sont fermés
            self.logic.proc.closeReadChannel(qt.QProcess.StandardOutput)
            self.logic.proc.closeReadChannel(qt.QProcess.StandardError)
            self.logic.proc.closeWriteChannel()
            
            logging.info("Process cleanup completed")
        
        # Important : supprimer la référence au processus
        if hasattr(self, "logic"):
            self.logic.proc = None
        
        # Déconnexion de tous les signaux et observers
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        # if self._parameterNode:
        #     self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        #     self._parameterNodeGuiTag = None
        #     self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        # self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        # if self.parent.isEntered:
        #     self.initializeParameterNode()

    def onPromptTextChanged(self) -> None:
        """Called when the prompt text is changed."""
        if self.ui.prompt.toPlainText():
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.enabled = False

    def onThinkBoxToggled(self, checked):
        self.logic.setThinking(checked)
        print(checked)


    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        # self.setParameterNode(self.logic.getParameterNode())

        # # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[SlicerGPTParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self.ui.prompt.plainText.selectAll() != '':
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            text = self.ui.prompt.toPlainText()
            self.ui.prompt.clear()
            message = {"role": "user", "content": text}
            self.ui.applyButton.enabled = False
            dialogue = self.logic.process(message)
            self.ui.conversation.setText(dialogue)

    def updateConversation(self, dialogue_text):
        """Met à jour l'interface utilisateur avec le nouveau dialogue.
        Cette méthode sera appelée par la logique quand une réponse asynchrone est reçue.
        """
        self.ui.conversation.setText(dialogue_text)
        
        # Réactiver le bouton d'envoi
        self.ui.applyButton.enabled = bool(self.ui.prompt.toPlainText())
    
    @staticmethod
    def areDependenciesSatisfied():
        from Scripts.PythonDependenciesManager import PythonDependencyChecker
        return PythonDependencyChecker.areDependenciesSatisfied()
    
    @staticmethod
    def downloadDependenciesAndRestart():
        from Scripts.PythonDependenciesManager import PythonDependencyChecker
        progressDialog = slicer.util.createProgressDialog(maximum=0)
        # extensionManager = slicer.app.extensionsManagerModel()

        # def downloadWithMetaData(extName):
        # # Method for downloading extensions prior to Slicer 5.0.3
        #     meta_data = extensionManager.retrieveExtensionMetadataByName(extName)
        #     if meta_data:
        #         return extensionManager.downloadAndInstallExtension(meta_data["extension_id"])

        # def downloadWithName(extName):
        # # Direct extension download since Slicer 5.0.3
        #     return extensionManager.downloadAndInstallExtensionByName(extName)

        # Install Slicer extensions
        # downloadF = downloadWithName if hasattr(extensionManager,
        #                                         "downloadAndInstallExtensionByName") else downloadWithMetaData

        # slicerExtensions = ["PyTorch"]
        # for slicerExt in slicerExtensions:
        #     progressDialog.labelText = f"Installing the {slicerExt}\nSlicer extension"
        #     downloadF(slicerExt)

        PythonDependencyChecker.installDependenciesIfNeeded(progressDialog)
        progressDialog.close()

        # Restart if no extension failed to download. Otherwise warn the user about the failure.
        # failedDownload = [slicerExt for slicerExt in slicerExtensions if
        #                 not extensionManager.isExtensionInstalled(slicerExt)]

        # if failedDownload:
        #     failed_ext_list = "\n".join(failedDownload)
        #     warning_msg = f"The download process failed install the following extensions : {failed_ext_list}" \
        #                     f"\n\nPlease try to manually install them using Slicer's extension manager"
        #     qt.QMessageBox.warning(None, "Failed to download extensions", warning_msg)
        # else:
        slicer.app.restart()

            


#
# SlicerGPTLogic
#

import requests
import sys
import re
from Scripts.Utils import extract_mrml_scene_as_text

class SlicerGPTLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.dialogue = []
        self.proc = qt.QProcess()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir not in sys.path:
            sys.path.append(base_dir)
        server_path = os.path.join(base_dir, "SlicerGPT", "Scripts", "LocalServer.py")
        self.proc.setProgram("pythonSlicer")
        self.proc.setArguments([server_path])

        self.proc.readyReadStandardOutput.connect(self.handle_stdout)
        self.proc.readyReadStandardError.connect(self.handle_stderr)
        self.start()
        self.proc.started.connect(lambda: print("[INFO] Server started"))
        self.proc.finished.connect(lambda: print("[INFO] Server stopped"))

        from Scripts.AsyncRequest import AsyncRequest
        self.async_request = AsyncRequest()
        self.async_request.requestFinished.connect(self.handleResponse)
        self.async_request.requestFailed.connect(self.handleError)

        self.widget = None

        
    def getParameterNode(self):
        return SlicerGPTParameterNode(super().getParameterNode())
    
    def start(self):
        print("[INFO] Starting process...")
        self.proc.start()

    def handle_stdout(self):
        output = self.proc.readAllStandardOutput().data().decode()
        print("[STDOUT]", output)

    def handle_stderr(self):
        raw = self.proc.readAllStandardError().data()
        error = raw.decode(errors="replace")
        print("[STDERR]", error)

    def handleResponse(self, response_data):
        """Gère la réponse reçue de façon asynchrone"""
        # Ajouter la réponse à notre dialogue
        print(response_data)
        self.dialogue.pop()
        self.dialogue.append({"role": "assistant", "content": response_data})
        
        # Mettre à jour l'interface utilisateur
        if self.widget:
            self.widget.updateConversation(self.formatDialogue())
            
    def handleError(self, error_message):
        """Gère les erreurs lors de l'appel asynchrone"""
        # Ajouter un message d'erreur à notre dialogue
        self.dialogue.append({"role": "assistant", "content": f"Erreur de communication avec le serveur: {error_message}"})
        
        # Mettre à jour l'interface utilisateur
        if self.widget:
            self.widget.updateConversation(self.formatDialogue())
    
    def setThinking(self, think):
        """
        Change the chatbot thinking mode.
        """
        requests.post("http://127.0.0.1:81/setThink", json={"think": think})
    
    def formatDialogue(self) -> str:
        """
        Return the formatted text of the dialogue, it will be displayed in the conversation widget.
        """
        finalDialogue = []
        for message in self.dialogue:
            content = message["content"].replace('\n', '<br>')
            content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'<think>(.+?)</think>', r'<i>\1</i>', content, flags=re.DOTALL)
            if message["role"] == "assistant":
                finalDialogue.append(f'<div style="text-align:left; margin: 5px;"><span style="color:red; font-weight:bold;">SlicerGPT:</span><br>{content}</div>')
            elif message["role"] == "user":
                finalDialogue.append(f'<div style="text-align:right; margin: 5px;"><span style="color:blue; font-weight:bold;">You:</span><br>{content}</div>')


        return "\n\n".join(finalDialogue)


    def process(self, message) -> str:
        """
        Run the processing algorithm.
        TODO complete when the final process will be planned.
        """

        import time

        startTime = time.time()
        logging.info("Processing started")

        self.dialogue.append(message)
        temp_message = {"role": "assistant", "content": "Traitement en cours..."}
        self.dialogue.append(temp_message)
        
        # Préparer le message avec les informations de la scène MRML
        message["mrml_scene"] = extract_mrml_scene_as_text()
        
        # Mise à jour initiale de l'interface avant l'envoi
        formatted_dialogue = self.formatDialogue()
        
        # Lancer la requête asynchrone
        self.async_request.post("http://127.0.0.1:81/generate", message)
        
        # Retourner le dialogue actuel avec l'indicateur de chargement
        return formatted_dialogue



#
# SlicerGPTTest
#


class SlicerGPTTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SlicerGPT1()

    def test_SlicerGPT1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SlicerGPT1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SlicerGPTLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
