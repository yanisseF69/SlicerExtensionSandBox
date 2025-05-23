import requests
from threading import Thread
import qt

class AsyncRequest(qt.QObject):
    """Class used to send HTTP asynchronous request."""
    # Define signals that will be emitted when the request is finished
    requestFinished = qt.Signal(dict)
    requestFailed = qt.Signal(str)

    def __init__(self):
        super().__init__()
        
    def post(self, url, json_data):
        """Run the asynchronous post request.
        
        Args:
            url: request's URL
            json_data: JSON data to send
            
        La réponse sera émise via le signal requestFinished.
        Les erreurs seront émises via le signal requestFailed.
        """
        # Create a thread to execute the request
        thread = Thread(target=self._execute_request, args=(url, json_data))
        thread.daemon = True  # The thread will terminate when the main program terminates
        thread.start()
    
    def _execute_request(self, url, json_data):
        """Method executed in a separate thread."""
        try:
            response = requests.post(url, json=json_data)
            response.raise_for_status()  # Raises an exception if the status is not 2xx
            
            # Process the response
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Use moveToThread's thread to emit the signal safely
                    qt.QApplication.instance().postEvent(
                        self, 
                        _CustomEvent(_CustomEvent.Success, data)
                    )
                except ValueError:
                    # Handle plain text response
                    qt.QApplication.instance().postEvent(
                        self, 
                        _CustomEvent(_CustomEvent.Success, {"content": response.text})
                    )
            else:
                error_msg = f"HTTP Error: {response.status_code}"
                qt.QApplication.instance().postEvent(
                    self, 
                    _CustomEvent(_CustomEvent.Error, error_msg)
                )
                
        except requests.exceptions.RequestException as e:
            # In case of error, emit the error signal
            error_msg = f"Request error: {str(e)}"
            qt.QApplication.instance().postEvent(
                self, 
                _CustomEvent(_CustomEvent.Error, error_msg)
            )

    # Override event method to handle our custom events
    def event(self, event):
        if event.type() == _CustomEvent.EventType:
            if event.event_kind == _CustomEvent.Success:
                self.requestFinished.emit(event.data)
            elif event.event_kind == _CustomEvent.Error:
                self.requestFailed.emit(event.data)
            return True
        return qt.QObject.event(self, event)


# Custom event class to safely pass data between threads
class _CustomEvent(qt.QEvent):
    # Define custom event type
    EventType = qt.QEvent.Type(qt.QEvent.registerEventType())
    
    # Event kinds
    Success = 0
    Error = 1
    
    def __init__(self, event_kind, data):
        super().__init__(_CustomEvent.EventType)
        self.event_kind = event_kind
        self.data = data