# Docker Compose v1.17.x Format Guide

## Key Differences from Later Versions

1. **No Version Key**
   - Do NOT include `version: '2.x'` or `version: '3.x'`
   - Start directly with service definitions

2. **Build Configuration**
   ```yaml
   # Correct v1 format:
   myservice:
     build: .
     dockerfile: Dockerfile.name
   
   # NOT supported in v1:
   myservice:
     build:
       context: .
       dockerfile: Dockerfile.name
   ```

3. **Network Mode**
   ```yaml
   # Correct v1 format:
   myservice:
     net: host
   
   # NOT supported in v1:
   myservice:
     network_mode: host
   ```

4. **Logging Configuration**
   ```yaml
   # Correct v1 format:
   myservice:
     log_driver: "json-file"
     log_opt:
       max-size: "100m"
   
   # NOT supported in v1:
   myservice:
     logging:
       driver: "json-file"
       options:
         max-size: "100m"
   ```

5. **Environment Variables**
   ```yaml
   # Correct v1 format:
   myservice:
     environment:
       - VARIABLE=value
   
   # Simple variable substitution only
   # NO default values with :-
   ```

6. **Indentation Rules**
   - All service properties must be indented with exactly 2 spaces
   - Each property must be on its own line
   - No inline property definitions

## Common Problems

1. **Incorrect Service Name Format**
   ```yaml
   # WRONG:
   servicename: other-stuff
   
   # Correct:
   servicename:
     other-stuff
   ```

2. **Property Alignment**
   ```yaml
   # WRONG:
   service:
   property: value
   
   # Correct:
   service:
     property: value
   ```

3. **Multiple Properties on One Line**
   ```yaml
   # WRONG:
   service: property1: value1  property2: value2
   
   # Correct:
   service:
     property1: value1
     property2: value2
   ```

## Example Valid v1 Configuration

```yaml
web:
  build: .
  dockerfile: Dockerfile
  container_name: webapp
  command: python app.py
  ports:
    - "8000:8000"
  environment:
    - DEBUG=1
    - API_KEY=secret
  volumes:
    - .:/code
  net: host
  log_driver: "json-file"
  log_opt:
    max-size: "100m"
```
