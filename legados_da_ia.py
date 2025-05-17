%pip -q install google-adk google-genai  # Instala as bibliotecas necessárias

import os
from google.colab import userdata
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from datetime import date
import textwrap
from IPython.display import display, Markdown
import warnings

warnings.filterwarnings("ignore")

# Configura a API Key do Google Gemini
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])   # Configura a API Key

MODEL_ID = "gemini-2.0-flash"   # Modelo para busca e planejamento (mais rápido)
MODEL_REDATOR_REVISOR = "gemini-2.5-pro-preview-03-25"  # Modelo para redação e revisão (mais capaz, se tiver tempo)

# Função auxiliar para chamar os agentes
def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text + "\n"
    return final_response

# Função auxiliar para formatar a saída
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Agente Buscador (Modificado)
def agente_buscador(figura_historica, data_de_hoje):
    buscador = Agent(
        name="agente_buscador",
        model=MODEL_ID,
        instruction=f"""
        Você é um historiador e pesquisador. Sua tarefa é usar a ferramenta de busca do Google (google_search)
        para encontrar informações relevantes sobre a vida, realizações e características de {figura_historica}.
        Foque em informações que ajudem a entender como a IA poderia ser relevante para essa pessoa.
        """,
        description="Agente que busca informações sobre figuras históricas",
        tools=[google_search]
    )
    entrada_do_agente = f"Figura Histórica: {figura_historica}\nData de hoje: {data_de_hoje}"
    informacoes = call_agent(buscador, entrada_do_agente)
    return informacoes

# Agente Planejador (Modificado)
def agente_planejador(figura_historica, informacoes_historicas):
    planejador = Agent(
        name="agente_planejador",
        model=MODEL_ID,
        instruction=f"""
        Você é um especialista em tecnologia e história. Com base nas informações sobre {figura_historica},
        sua tarefa é planejar como essa pessoa poderia utilizar a IA hoje.
        Considere seus talentos, áreas de interesse e o impacto que a IA poderia ter em seu trabalho e legado.
        """,
        description="Agente que planeja a interação da figura histórica com a IA",
        tools=[]   # Não precisa de ferramentas
    )
    entrada_do_agente = f"Figura Histórica: {figura_historica}\nInformações Históricas: {informacoes_historicas}"
    plano = call_agent(planejador, entrada_do_agente)
    return plano

# Agente Redator (Modificado)
def agente_redator(figura_historica, plano_de_ia):
    redator = Agent(
        name="agente_redator",
        model=MODEL_REDATOR_REVISOR,
        instruction=f"""
        Você é um escritor criativo. Com base no plano sobre como {figura_historica} poderia usar a IA,
        escreva um texto envolvente e informativo sobre o tema.
        Use uma linguagem clara e acessível, imaginando como essa pessoa se adaptaria ao mundo da IA.
        """,
        description="Agente que escreve o texto sobre a figura histórica e a IA"
    )
    entrada_do_agente = f"Figura Histórica: {figura_historica}\nPlano de IA: {plano_de_ia}"
    texto = call_agent(redator, entrada_do_agente)
    return texto

# Agente Revisor (Opcional - Se tiver tempo)
def agente_revisor(figura_historica, texto_gerado):
    revisor = Agent(
        name="agente_revisor",
        model=MODEL_REDATOR_REVISOR,
        instruction=f"""
        Você é um revisor de texto. Revise o texto sobre {figura_historica} e a IA,
        verificando clareza, concisão, correção e estilo.
        Sugira melhorias se necessário.
        """,
        description="Agente que revisa o texto"
    )
    entrada_do_agente = f"Figura Histórica: {figura_historica}\nTexto Gerado: {texto_gerado}"
    texto_revisado = call_agent(revisor, entrada_do_agente)
    return texto_revisado

# --- Fluxo Principal ---
data_de_hoje = date.today().strftime("%d/%m/%Y")

print("🚀 Iniciando Legados da IA 🚀")

figura = input("Digite o nome de uma figura histórica: ")

if figura:
    print(f"\nBuscando informações sobre {figura}...\n")
    informacoes_historicas = agente_buscador(figura, data_de_hoje)
    print("\nPlanejando como a IA poderia ser usada...\n")
    plano_de_ia = agente_planejador(figura, informacoes_historicas)
    print("\nGerando o texto...\n")
    texto_final = agente_redator(figura, plano_de_ia)

    # Se tiver tempo, adicione a revisão
    # texto_final = agente_revisor(figura, texto_final)

    print("\nResultado:\n")
    display(to_markdown(texto_final))

    # --- Preparar para o GitHub ---
    # 1. Salve este código como legados_da_ia.py
    # 2. Crie um arquivo README.md (conforme a estrutura que te dei)
    # 3. Suba ambos para o GitHub
    # 4. Submeta o link do GitHub no formulário!

else:
    print("Nenhuma figura histórica fornecida.")