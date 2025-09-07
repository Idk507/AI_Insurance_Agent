import msal
import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# MSAL Configuration from environment variables with fallbacks
CLIENT_ID = os.getenv("MSAL_CLIENT_ID", "f8df195a-151f-44df-88ce-215640d63afe")
TENANT_ID = os.getenv("MSAL_TENANT_ID", "9f2e2b7d-6081-4b3a-b7f5-433b581f6a5f")
CLIENT_SECRET = os.getenv("MSAL_CLIENT_SECRET", "MGv8Q~.B3LGdDt9bmqYAcoLtRXHQ3hyzudOm7a_V")
AUTHORITY = os.getenv("MSAL_AUTHORITY", f"https://login.microsoftonline.com/{TENANT_ID}")

# Validate configuration
if not all([CLIENT_ID, TENANT_ID, CLIENT_SECRET]):
    logger.error("One or more MSAL environment variables are not set.")
    # Continue with hardcoded values for testing
    logger.info("Using hardcoded MSAL values for testing purposes.")

# Scopes for Microsoft Graph API
SCOPE = ["User.Read", "User.ReadBasic.All"]

# Create a Confidential Client Application
app = msal.ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET
)

def initiate_auth_flow(redirect_uri: str, state: Optional[str] = None) -> Dict[str, Any]:
    """Initiate the authorization code flow and return the flow details."""
    logger.info(f"Initiating auth flow with redirect_uri: {redirect_uri}")
    
    flow = app.initiate_auth_code_flow(
        scopes=SCOPE,
        redirect_uri=redirect_uri,
        state=state
    )
    
    logger.info("Authorization flow initiated successfully")
    logger.debug(f"Flow details: {flow}")
    return flow

def build_auth_url(redirect_uri: str, state: Optional[str] = None) -> str:
    """Build the authorization URL for Azure AD login using newer MSAL method."""
    logger.info(f"Building auth URL with redirect_uri: {redirect_uri}")
    
    flow = initiate_auth_flow(redirect_uri, state)
    auth_url = flow.get("auth_uri")
    
    if not auth_url:
        logger.error("Failed to generate authorization URL")
        raise Exception("Failed to generate authorization URL")
    
    logger.info(f"Generated auth URL: {auth_url}")
    return auth_url

def acquire_token_by_auth_code_flow(auth_response: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
    """Acquire token using the newer auth code flow method."""
    logger.info("Acquiring token using auth code flow")
    
    try:
        result = app.acquire_token_by_auth_code_flow(flow, auth_response)
        
        if "access_token" in result:
            logger.info("Token acquired successfully using auth code flow")
            logger.debug(f"Token result keys: {list(result.keys())}")
            return result
        else:
            logger.error(f"Token acquisition failed: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Exception during token acquisition: {str(e)}")
        return {"error": "token_acquisition_failed", "error_description": str(e)}

def acquire_token_by_auth_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """Acquire token using authorization code (fallback method for compatibility)."""
    logger.info("Acquiring token using authorization code (fallback method)")
    logger.warning("Using deprecated method - consider updating to auth_code_flow")
    
    try:
        result = app.acquire_token_by_authorization_code(
            code=code,
            scopes=SCOPE,
            redirect_uri=redirect_uri
        )
        
        if "access_token" in result:
            logger.info("Token acquired successfully using authorization code")
            logger.debug(f"Token result keys: {list(result.keys())}")
            return result
        else:
            logger.error(f"Token acquisition failed: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Exception during token acquisition: {str(e)}")
        return {"error": "token_acquisition_failed", "error_description": str(e)}

def get_token_from_cache(account: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Get token from cache if available."""
    logger.info("Attempting to get token from cache")
    
    try:
        accounts = app.get_accounts(username=account.get("username") if account else None)
        logger.debug(f"Found {len(accounts)} accounts in cache")
        
        if accounts:
            # Use the first account
            selected_account = accounts[0]
            logger.info(f"Using account: {selected_account.get('username', 'Unknown')}")
            
            result = app.acquire_token_silent(SCOPE, account=selected_account)
            
            if "access_token" in result:
                logger.info("Token acquired from cache successfully")
                return result
            else:
                logger.info("No valid token found in cache")
                return None
        else:
            logger.info("No accounts found in cache")
            return None
            
    except Exception as e:
        logger.error(f"Exception getting token from cache: {str(e)}")
        return None

def remove_account(account: Dict[str, Any]) -> bool:
    """Remove account from MSAL cache."""
    logger.info("Removing account from MSAL cache")
    
    try:
        if account:
            app.remove_account(account)
            logger.info("Account removed from cache successfully")
            return True
        else:
            logger.warning("No account provided to remove")
            return False
            
    except Exception as e:
        logger.error(f"Exception removing account from cache: {str(e)}")
        return False

def acquire_token():
    """Acquire a token from Azure AD using client credentials."""
    logger.info("Acquiring token using client credentials flow")
    
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" in result:
        logger.info("Token acquired successfully using client credentials")
        return result["access_token"]
    else:
        logger.error("Failed to acquire token using client credentials")
        logger.error(f"Error: {result.get('error')}")
        logger.error(f"Error description: {result.get('error_description')}")
        return None

if __name__ == "__main__":
    # Example usage
    logger.info("Testing MSAL authentication...")
    token = acquire_token()
    if token:
        logger.info("Authentication test successful!")
        print(f"Token: {token[:50]}...")
    else:
        logger.error("Authentication test failed!")  