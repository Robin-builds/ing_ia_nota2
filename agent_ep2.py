"""
=============================================================================
IACC Analytics Agent - Evaluacion Parcial N 2
Asignatura : ISY0101 - Ingenieria de Soluciones con IA
Estudiante : Robinson Arriagada Borquez
Modulos    : IL2.1 (herramientas), IL2.2 (memoria), IL2.3 (planificacion)
=============================================================================

Arquitectura del agente:

  Interfaz CLI  ->  IACCAnalyticsAgent
    - Memoria corto plazo : ConversationBufferWindowMemory (k=5)
    - Memoria semantica   : FAISS local (cumple Ley 19.628)
    - Herramienta 1       : faiss_retriever
    - Herramienta 2       : calcular_kpi
    - Herramienta 3       : clasificar_consulta
    - Planificador        : ReAct via LangChain AgentExecutor

Privacidad: los embeddings y el indice FAISS permanecen en disco local.
El LLM recibe unicamente contexto anonimizado; ningun dato nominal sale
hacia servicios externos (Ley 19.628).
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS Y CONFIGURACION DE ENTORNO
# ---------------------------------------------------------------------------
import os
import json
import datetime
from typing import Optional

# LangChain core
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub

# RAG / Vector store
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Variables de entorno requeridas:
#   GITHUB_TOKEN    = token personal de GitHub (ghp_...)
#   GITHUB_BASE_URL = https://models.inference.ai.azure.com
os.environ.setdefault("OPENAI_API_KEY",  os.getenv("GITHUB_TOKEN", ""))
os.environ.setdefault("OPENAI_API_BASE", os.getenv(
    "GITHUB_BASE_URL", "https://models.inference.ai.azure.com"))


# ---------------------------------------------------------------------------
# 2. BASE DE CONOCIMIENTO IACC (datos sinteticos anonimizados)
#    En produccion estos chunks provienen de ingest.py + preprocess.py
#    sobre la exportacion real de HubSpot CRM.
# ---------------------------------------------------------------------------
IACC_KNOWLEDGE_BASE = """
PERIODO: PEM-2025-01 | Leads totales: 1240 | Matriculas: 186 | Tasa conversion: 15%
Equipo Grupo Pro - Leads: 620 | Matriculas: 112 | Tasa: 18.1%
Equipo Grupo Aleatorio - Leads: 620 | Matriculas: 74 | Tasa: 11.9%

Top carreras por matricula PEM-2025-01:
1. Administracion de Empresas - 42 matriculas
2. Contabilidad General - 38 matriculas
3. Tecnico en Enfermeria - 31 matriculas
4. Ingenieria en Informatica - 28 matriculas
5. Prevencion de Riesgos - 22 matriculas

Escuela Administracion y Negocios: 80 matriculas (43%)
Escuela Salud: 53 matriculas (28.5%)
Escuela Informatica y Telecomunicaciones: 28 matriculas (15%)
Escuela Construccion: 25 matriculas (13.5%)

Region con mas leads: Metropolitana (RM) - 487 leads (39.3%)
Region "No informado": 86% de los registros (problema de calidad en CRM)

PERIODO: PEM-2024-02 | Leads totales: 1105 | Matriculas: 152 | Tasa conversion: 13.8%
Variacion inter-periodo: +12.3% matriculas respecto a PEM-2024-02

Definicion KPI - Tasa de Conversion: Matriculas / Leads * 100
Definicion KPI - Velocidad de Cierre: dias promedio desde primer contacto hasta matricula
Velocidad de cierre promedio PEM-2025-01: 18 dias
Velocidad de cierre Grupo Pro: 14 dias | Grupo Aleatorio: 23 dias

Politica de privacidad: RUT, nombre y datos de contacto excluidos del sistema (Ley 19.628)
"""


# ---------------------------------------------------------------------------
# 3. CONSTRUCCION DEL INDICE FAISS (memoria semantica de largo plazo)
# ---------------------------------------------------------------------------
def construir_indice_faiss() -> FAISS:
    """
    Genera el indice vectorial FAISS desde la base de conocimiento institucional.
    En produccion se carga directamente desde disco (faiss_index/).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        length_function=len
    )
    chunks = splitter.split_text(IACC_KNOWLEDGE_BASE)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


# ---------------------------------------------------------------------------
# 4. HERRAMIENTAS DEL AGENTE (IL2.1)
# ---------------------------------------------------------------------------

# El vectorstore se construye una sola vez por sesion (patron singleton)
_vectorstore: Optional[FAISS] = None


def _get_vectorstore() -> FAISS:
    """Devuelve el vectorstore existente o lo construye si no existe."""
    global _vectorstore
    if _vectorstore is None:
        print("Construyendo indice FAISS...")
        _vectorstore = construir_indice_faiss()
        print("Indice FAISS listo.\n")
    return _vectorstore


@tool
def faiss_retriever(consulta: str) -> str:
    """
    Busca informacion analitica sobre admision IACC en la base de conocimiento
    vectorial local (FAISS). Usar para consultas sobre leads, matriculas,
    tasas de conversion, equipos comerciales, carreras o periodos.
    Argumentos:
        consulta: pregunta en lenguaje natural sobre datos de admision IACC.
    """
    vs = _get_vectorstore()
    # k dinamico: ampliar recuperacion para consultas de equipo completo
    k = 6 if any(w in consulta.lower() for w in ["todos", "equipo", "completo", "todas"]) else 3
    docs = vs.similarity_search(consulta, k=k)
    if not docs:
        return "No se encontro informacion relevante para esa consulta."
    contexto = "\n---\n".join(d.page_content for d in docs)
    return f"[CONTEXTO RECUPERADO - {len(docs)} chunks]\n{contexto}"


@tool
def calcular_kpi(leads: float, matriculas: float, dias_cierre: Optional[float] = None) -> str:
    """
    Calcula KPIs de admision a partir de valores numericos.
    Usar cuando el usuario solicita calcular tasa de conversion o velocidad de cierre.
    Argumentos:
        leads       : numero total de leads en el periodo.
        matriculas  : numero total de matriculas confirmadas.
        dias_cierre : promedio de dias hasta matricula (opcional).
    """
    if leads <= 0:
        return "Error: el numero de leads debe ser mayor que 0."

    tasa = round((matriculas / leads) * 100, 2)
    resultado = {
        "tasa_conversion": f"{tasa}%",
        "formula": "Matriculas / Leads x 100",
        "leads": int(leads),
        "matriculas": int(matriculas),
    }
    if dias_cierre is not None:
        resultado["velocidad_cierre_dias"] = dias_cierre
        resultado["clasificacion"] = (
            "Alta eficiencia (menos de 15 dias)" if dias_cierre < 15
            else "Eficiencia media (15 a 25 dias)" if dias_cierre <= 25
            else "Requiere mejora (mas de 25 dias)"
        )
    return json.dumps(resultado, ensure_ascii=False, indent=2)


@tool
def clasificar_consulta(consulta: str) -> str:
    """
    Clasifica el tipo de consulta y define la estrategia de respuesta.
    Usar SIEMPRE como primer paso antes de responder cualquier pregunta.
    Argumentos:
        consulta: pregunta original del usuario en lenguaje natural.
    """
    c = consulta.lower()

    if any(w in c for w in ["calcula", "calculo", "compute", "kpi"]):
        tipo   = "CALCULO_KPI"
        accion = "Usar calcular_kpi con los valores provistos o recuperados del contexto."
    elif any(w in c for w in ["compara", "diferencia", "versus", "vs", "mejor"]):
        tipo   = "COMPARACION"
        accion = "Recuperar datos de ambos periodos o equipos con faiss_retriever y contrastar."
    elif any(w in c for w in ["todos", "equipo", "completo", "carrera", "escuela"]):
        tipo   = "CONSULTA_AMPLIA"
        accion = "Usar faiss_retriever con k ampliado para cubrir todos los registros relevantes."
    elif any(w in c for w in ["periodo", "historico", "tendencia", "variacion"]):
        tipo   = "ANALISIS_TEMPORAL"
        accion = "Recuperar datos de multiples periodos con faiss_retriever y calcular variaciones."
    else:
        tipo   = "CONSULTA_PUNTUAL"
        accion = "Usar faiss_retriever con la consulta directa."

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return (
        f"[CLASIFICACION - {timestamp}]\n"
        f"Tipo    : {tipo}\n"
        f"Accion  : {accion}\n"
        f"Nota    : Responder solo con datos del contexto recuperado. "
        f"No inventar cifras. Citar el periodo en cada dato."
    )


# Registro de herramientas disponibles para el agente
TOOLS = [clasificar_consulta, faiss_retriever, calcular_kpi]


# ---------------------------------------------------------------------------
# 5. CLASE PRINCIPAL DEL AGENTE (IL2.1 + IL2.2 + IL2.3)
# ---------------------------------------------------------------------------
class IACCAnalyticsAgent:
    """
    Agente conversacional de analitica de admision para IACC.

    Componentes:
    - LLM           : gpt-4o via GitHub Models API
    - Herramientas  : faiss_retriever, calcular_kpi, clasificar_consulta
    - Memoria CP    : ConversationBufferWindowMemory (ultimas 5 interacciones)
    - Memoria LP    : FAISS vectorstore (busqueda semantica local)
    - Planificacion : patron ReAct (Reason + Act) via LangChain
    """

    def __init__(self, verbose: bool = True):
        # Modelo de lenguaje
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=1000
        )

        # Memoria de corto plazo (IL2.2)
        # Ventana de 5 turnos: suficiente para multi-turno sin exceder el limite de tokens
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )

        # Prompt ReAct estandar desde LangChain Hub
        # Incluye los slots: {tools}, {tool_names}, {chat_history}, {agent_scratchpad}
        self.prompt = hub.pull("hwchase17/react-chat")

        # Construccion del agente ReAct
        agent = create_react_agent(
            llm=self.llm,
            tools=TOOLS,
            prompt=self.prompt
        )

        # Executor: orquesta el ciclo razonamiento -> accion -> observacion
        self.executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=self.memory,
            verbose=verbose,           # muestra el razonamiento ReAct paso a paso
            handle_parsing_errors=True,
            max_iterations=6           # limite de pasos para evitar bucles
        )

        # Contexto de sistema inyectado en cada consulta
        self._system_context = (
            "Eres IACC Analytics Assistant, especializado en analitica de gestion "
            "comercial de admision. Responde unicamente con datos recuperados del contexto. "
            "Nunca reveles RUT, nombre ni contacto de estudiantes (Ley 19.628). "
            "Formato de respuesta: tabla o lista con valor destacado y periodo citado."
        )

    def consultar(self, pregunta: str) -> str:
        """
        Procesa una consulta e invoca el ciclo ReAct del agente.
        El agente decide autonomamente que herramientas usar y en que orden.
        """
        entrada = f"{self._system_context}\n\nConsulta del analista: {pregunta}"
        try:
            resultado = self.executor.invoke({"input": entrada})
            return resultado.get("output", "Sin respuesta disponible.")
        except Exception as e:
            return f"Error en el agente: {e}"

    def limpiar_memoria(self):
        """Reinicia el historial de conversacion para iniciar un nuevo contexto."""
        self.memory.clear()
        print("Memoria de conversacion reiniciada.")


# ---------------------------------------------------------------------------
# 6. INTERFAZ CLI
# ---------------------------------------------------------------------------
def main():
    """
    Sesion interactiva de consultas analiticas IACC.
    Comandos:
        /limpiar  -> reinicia el historial de conversacion
        /salir    -> cierra la sesion
    """
    print("=" * 60)
    print("  IACC Analytics Assistant - EP2 (ISY0101)")
    print("  Ingresa consultas en lenguaje natural.")
    print("  /limpiar = nueva sesion  |  /salir = terminar")
    print("=" * 60)

    agente = IACCAnalyticsAgent(verbose=True)
    _get_vectorstore()  # pre-carga el indice al inicio

    while True:
        try:
            pregunta = input("\nConsulta > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSesion finalizada.")
            break

        if not pregunta:
            continue
        if pregunta.lower() == "/salir":
            print("Sesion finalizada.")
            break
        if pregunta.lower() == "/limpiar":
            agente.limpiar_memoria()
            continue

        print("\n" + "-" * 50)
        respuesta = agente.consultar(pregunta)
        print("\nRespuesta:\n")
        print(respuesta)
        print("-" * 50)


if __name__ == "__main__":
    main()
