from supabase import create_client, Client
from rag.config.settings import settings

def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client instance using environment variables.
    
    Returns:
        Client: Configured Supabase client instance
    """
    try:
        client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Supabase client: {e}")

# Create a singleton instance
supabase_client = get_supabase_client() 