from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"

    # Retrieval Neo4j (exposed through an MCP server)
    retrieval_neo4j_uri: str = "neo4j+s://demo.neo4jlabs.com:7687"
    retrieval_neo4j_database: str = "companies2"
    retrieval_neo4j_username: str = "companies2"
    retrieval_neo4j_password: str = "companies2"

    # Memory Neo4j (local docker)
    memory_neo4j_uri: str = "bolt://localhost:7687"
    memory_neo4j_username: str = "neo4j"
    memory_neo4j_password: str = "password"
    memory_neo4j_database: str = "neo4j"


settings = Settings()
