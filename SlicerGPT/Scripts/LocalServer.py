import os
import signal
import time
import logging
import sys
import threading
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
from typing import Any, Optional
from Model import Model
from VectorStoreManager import VectorStoreManager

# Configuration des logs
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("server")

class Message(BaseModel):
    role: str
    content: str
    mrml_scene: Optional[str] = None

class ThinkBool(BaseModel):
    think: bool


inferenceServer = FastAPI()

# Variables globales pour le serveur
server_should_exit = False
server_pid = os.getpid()


# Préchargement du modèle et de la base vectorielle au démarrage
logger.info("Initializing vector store and model...")
start_time = time.time()
base_dir = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(base_dir, "..", "Data", "SlicerFAISS")

# Initialiser le gestionnaire de base vectorielle et le modèle
manager = VectorStoreManager(faiss_path)
chatbot = Model(manager=manager)
logger.info(f"Initialization complete in {time.time() - start_time:.2f} seconds")

@inferenceServer.post("/setThink")
async def setThink(think: ThinkBool):
    chatbot.enable_thinking = think.think


@inferenceServer.post("/generate")
async def generate(message: Message):
    # Log de réception
    logger.info("Starting generate function execution")
    start_time = time.time()
    
    # Générer la réponse
    try:
        response = chatbot.generate_response(message.content, message.mrml_scene)
        logger.info(f"generate function completed in {time.time() - start_time:.4f} seconds")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise


@inferenceServer.get("/health")
async def health_check():
    """Endpoint simple pour vérifier que le serveur est opérationnel"""
    return {"status": "ok", "timestamp": time.time()}


@inferenceServer.get("/shutdown")
async def shutdown():
    """Endpoint pour arrêter proprement le serveur"""
    logger.info("Shutdown request received")
    
    # Déclencher l'arrêt dans un thread séparé pour pouvoir renvoyer une réponse
    def stop_server():
        logger.info("Shutting down server...")
        time.sleep(0.5)  # Petit délai pour s'assurer que la réponse est envoyée
        global server_should_exit
        server_should_exit = True
        
        # Arrêter le processus lui-même
        os.kill(server_pid, signal.SIGTERM)
    
    threading.Thread(target=stop_server).start()
    return {"status": "shutting_down"}


# Fonction de démarrage pour le serveur Uvicorn
def run_server():
    logger.info(f"Starting server on port 81, PID: {server_pid}")
    
    # Configuration du serveur Uvicorn
    config = uvicorn.Config(
        app=inferenceServer, 
        host="127.0.0.1",
        port=81,
        log_level="info",
        loop="asyncio",
        workers=1
    )
    
    # Démarrer le serveur
    server = uvicorn.Server(config)
    server.run()


if __name__=="__main__":
    # Intercepter SIGTERM pour un arrêt propre
    def handle_sigterm(signum, frame):
        logger.info("SIGTERM received, shutting down")
        global server_should_exit
        server_should_exit = True
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Démarrer le serveur
    run_server()