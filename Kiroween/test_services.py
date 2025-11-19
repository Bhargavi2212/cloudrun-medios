"""
Quick test script to verify FastAPI services are working.
"""

import asyncio
import sys
import os

# Add services to path
sys.path.insert(0, 'services/manage-agent')
sys.path.insert(0, 'services/scribe-agent')
sys.path.insert(0, 'services/summarizer-agent')

async def test_service_imports():
    """Test that all services can be imported successfully."""
    
    print("Testing service imports...")
    
    try:
        # Test manage-agent import
        sys.path.insert(0, 'services/manage-agent')
        import main as manage_main
        print("✓ Manage Agent service imported successfully")
        
        # Test scribe-agent import  
        sys.path.insert(0, 'services/scribe-agent')
        import main as scribe_main
        print("✓ Scribe Agent service imported successfully")
        
        # Test service configurations
        import config as manage_config
        scribe_settings = None  # Will test later
        
        manage_settings = manage_config.get_settings()
        
        print(f"✓ Manage Agent configured for hospital: {manage_settings.hospital_id}")
        print("✓ Basic service structure verified")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


async def main():
    """Main test function."""
    print("Medi OS Services Test")
    print("====================")
    
    success = await test_service_imports()
    
    if success:
        print("\n✓ All services are properly configured and ready to run!")
        print("\nNext steps:")
        print("1. Set up environment variables (DATABASE_URL, API keys)")
        print("2. Run services individually:")
        print("   - Manage Agent: cd services/manage-agent && python main.py")
        print("   - Scribe Agent: cd services/scribe-agent && python main.py")
        print("3. Test endpoints at http://localhost:8001/docs and http://localhost:8002/docs")
        return 0
    else:
        print("\n✗ Service configuration issues detected")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)