# IACC Analytics Agent

Agente conversacional para automatizar la consulta analitica de admision en el Instituto Profesional IACC. El sistema permite realizar consultas en lenguaje natural sobre datos de HubSpot CRM, reduciendo el tiempo de generacion de reportes de horas a minutos.

**Asignatura:** ISY0101 - Ingenieria de Soluciones con IA  
**Escuela:** Informatica y Telecomunicaciones, DUOC UC  
**Docente:** Cristian Carcamo Mansilla  
**Estudiante:** Robinson Arriagada Borquez  

---

## Descripcion del problema

La Direccion de Admision de IACC requeria entre 2 y 4 horas por reporte complejo: exportar datos de HubSpot, cruzarlos en Excel, interpretarlos y formatearlos para gerencia. El proceso se repetia ante cada nueva consulta y dependia de un solo analista como cuello de botella.

Este proyecto implementa una solucion RAG + Agente que responde consultas analiticas en lenguaje natural, citando la fuente y el periodo en cada dato, sin exponer informacion nominal de postulantes.

---

## Arquitectura

```
Usuario (CLI)
    |
    v
IACCAnalyticsAgent
    |-- ConversationBufferWindowMemory  (memoria corto plazo, k=5)
    |
    v
AgentExecutor (LangChain ReAct)
    |           |            |
    v           v            v
clasificar   faiss        calcular
_consulta   _retriever     _kpi
              |
          FAISS local
          (indice vectorial)
              |
              v
     gpt-4o via GitHub Models API
              |
              v
     Respuesta + historial actualizado
```

**Componentes principales:**

| Componente | Herramienta | Razon de eleccion |
|---|---|---|
| Orquestacion | LangChain AgentExecutor | Abstrae el ciclo ReAct y el manejo de errores |
| LLM | gpt-4o (GitHub Models) | API gratuita compatible con OpenAI |
| Embeddings | text-embedding-3-small | Disponible via GitHub Models sin infra adicional |
| Vector store | FAISS local | Sin transmision de datos externos; cumple Ley 19.628 |
| Memoria CP | ConversationBufferWindowMemory | Ventana de 5 turnos sin overflow de contexto |
| Planificacion | ReAct (hwchase17/react-chat) | Razonamiento trazable paso a paso |

---

## Estructura del repositorio

```
/
|-- EP1 - Pipeline RAG
|   |-- preprocess.py          # Normalizacion del CSV de HubSpot
|   |-- ingest.py              # Generacion de embeddings e indice FAISS
|   |-- 1-basic-rag.ipynb      # RAG basico sobre texto plano
|   |-- 2-text-chunking.py     # Estrategias de chunking
|   |-- 3-embeddings-simple-rag.ipynb
|   |-- 4-vector-rag.ipynb     # Pipeline RAG completo con FAISS
|   |-- 1-evaluation-rag.py    # Evaluacion con RAGAS
|   `-- 2-langsmith-evaluation.ipynb
|
|-- EP2 - Agente funcional
|   |-- agent_ep2.py           # Agente principal (IL2.1, IL2.2, IL2.3)
|   `-- demo_ep2.ipynb         # Demostracion de los tres escenarios
|
|-- Modulos del curso (IL1 a IL3)
|   |-- 1-github_model_api.ipynb
|   |-- 2-langchain_model_api.ipynb
|   |-- ... (notebooks por modulo)
|
|-- Informes
|   |-- Informe_EP1_ISY0101_Robinson_Arriagada.docx
|   `-- Informe_EP2_ISY0101_Robinson_Arriagada.docx
|
`-- README.md
```

---

## Requisitos

- Python 3.10 o superior
- Token de GitHub con acceso a GitHub Models
- Las dependencias listadas en `requirements.txt`

### Instalacion de dependencias

```bash
# Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

`requirements.txt` minimo para EP2:

```
langchain
langchain-openai
langchain-community
faiss-cpu
openai
```

---

## Configuracion de variables de entorno

El agente usa GitHub Models API como proveedor LLM. Se requieren dos variables:

```bash
# Windows (PowerShell)
$env:GITHUB_TOKEN    = "ghp_tu_token_aqui"
$env:GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

# Linux / macOS
export GITHUB_TOKEN="ghp_tu_token_aqui"
export GITHUB_BASE_URL="https://models.inference.ai.azure.com"
```

Para obtener un token: https://github.com/settings/tokens  
El token requiere el scope `read:packages` o simplemente ser un token clasico activo.

---

## Ejecucion

### Agente en modo interactivo (CLI)

```bash
python agent_ep2.py
```

El sistema construye el indice FAISS al iniciar y luego queda a la espera de consultas.

Ejemplos de consultas validas:

```
Consulta > Cual fue la tasa de conversion en PEM-2025-01?
Consulta > Compara el desempeno del Grupo Pro con el Grupo Aleatorio
Consulta > Calcula la tasa de conversion con 920 leads y 148 matriculas
Consulta > Cuales fueron las carreras con mas matriculas?
```

Comandos especiales:

```
/limpiar    reinicia el historial de conversacion
/salir      cierra la sesion
```

### Demostracion en notebook

Abrir `demo_ep2.ipynb` en Jupyter y ejecutar las celdas en orden. El notebook incluye tres escenarios documentados con el flujo ReAct esperado y las evidencias por indicador de evaluacion.

```bash
jupyter notebook demo_ep2.ipynb
```

---

## Herramientas del agente

El agente dispone de tres herramientas que selecciona autonomamente segun la consulta:

**`clasificar_consulta`**  
Analiza la intencion de la consulta y define la estrategia de respuesta. Se ejecuta siempre como primer paso del ciclo ReAct. Clasifica en: CONSULTA_PUNTUAL, CALCULO_KPI, COMPARACION, CONSULTA_AMPLIA, ANALISIS_TEMPORAL.

**`faiss_retriever`**  
Realiza busqueda semantica en el indice FAISS local. Recupera k=3 chunks para consultas puntuales y k=6 para consultas amplias o de comparacion.

**`calcular_kpi`**  
Calcula tasa de conversion (Matriculas / Leads * 100) y velocidad de cierre a partir de valores numericos provistos por el usuario.

---

## Decisiones de diseno relevantes

**Privacidad (Ley 19.628):** todos los embeddings y el indice FAISS se almacenan en disco local. El LLM recibe unicamente contexto agregado y anonimizado. Ningun RUT, nombre ni dato de contacto es indexado ni transmitido.

**k dinamico:** la herramienta `faiss_retriever` duplica el numero de chunks recuperados cuando detecta palabras como "todos" o "equipo completo", mejorando la cobertura sin aumentar el costo en consultas simples.

**Memoria de ventana:** `ConversationBufferWindowMemory` con k=5 permite resolver referencias implicitas entre turnos (por ejemplo, "cuantos dias tardaron" refiriendose a un equipo mencionado antes) sin exceder el limite de tokens del modelo.

**GitHub Models como proveedor LLM:** interfaz identica a la API de OpenAI, sin costo y sin necesidad de tarjeta de credito, lo que elimina la friccion de configuracion para entornos academicos.

---

## Evaluacion EP1 (pipeline RAG)

La calidad del pipeline RAG se mide con RAGAS sobre un dataset de 5 casos representativos del dominio IACC:

| Metrica | Umbral | Descripcion |
|---|---|---|
| Faithfulness | >= 0.85 | El LLM no genera datos ausentes en el contexto recuperado |
| Context Precision | - | Los chunks recuperados son relevantes para la consulta |
| Context Recall | - | Se recuperaron todos los chunks necesarios |
| Answer Relevancy | - | La respuesta responde efectivamente la pregunta |

---

## Problema conocido de calidad de datos

El 86% de los registros de la exportacion HubSpot tienen `region = "No informado"`. Este es un problema de completitud en el CRM de origen, no un error del pipeline. Se documenta aqui como hallazgo para el equipo de admision.

---

## Referencias

Chase, H. (2022). LangChain [Software]. https://github.com/langchain-ai/langchain

Es, E., Dinan, E., Lewis, P., & Riedel, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. arXiv:2309.15217.

Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. arXiv:2210.03629.
