# Environment Variables

- `ARK_ROOT`: The installation directory of ARK. Defaults to `/usr/local/ark` when unset.  

```  
export ARK_ROOT=/usr/local/ark  
```  

- `ARK_SCHEDULER`: The scheduler used by ARK. The available schedulers are `Default` and `Simple`. Defaults to `Default` when unset. `Simple` is a simple scheduler that is used for debugging.

```  
export ARK_SCHEDULER=Default  
```  

- `ARK_LOG_LEVEL`: The log level of ARK. The available log levels are `DEBUG`, `INFO`, `WARN`, and `ERROR`. Defaults to `INFO` when unset.

```
export ARK_LOG_LEVEL=DEBUG
```

- `ARK_IPC_LISTEN_PORT_BASE`: The base port number for IPC communication. Defaults to `42000` when unset. If we start multiple ARK processes on the same machine, different processes should use different port numbers. For example, if the base port is set to `42000`, the first process will use `42000`, the second process will use `42001`. Note that if the port number is already in use, ARK will fail to start.

```
export ARK_IPC_LISTEN_PORT_BASE=42000
```
