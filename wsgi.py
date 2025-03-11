from main import create_app
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a fresh instance of the app
app = create_app()

routes = [rule.rule for rule in app.url_map.iter_rules()]
logger.info(f"WSGI: Application created with routes: {routes}")
logger.info(f"WSGI: Registered blueprints: {list(app.blueprints.keys())}")

if __name__ == "__main__":
    app.run()