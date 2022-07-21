import pytest
import socket
import pyChainedProxy as socks


@pytest.fixture(scope='session')
def use_proxy():
    
    import socket
    import pyChainedProxy as socks

    chain = [
    'socks5://127.0.0.1:7891', 
    ]
    socks.setdefaultproxy() # Clear the default chain
    #adding hops with proxies
    for hop in chain:
        socks.adddefaultproxy(*socks.parseproxy(hop))

    # Configure alternate routes (No proxy for localhost)
    socks.setproxy('localhost', socks.PROXY_TYPE_NONE)
    socks.setproxy('127.0.0.1', socks.PROXY_TYPE_NONE)

    # Monkey Patching whole socket class (everything will be proxified)
    rawsocket = socket.socket
    socket.socket = socks.socksocket
