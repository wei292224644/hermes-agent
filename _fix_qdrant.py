from qdrant_client import QdrantClient
import shutil
import os

# Delete old collection at /tmp/qdrant
client = QdrantClient(path='/tmp/qdrant')
for c in client.get_collections().collections:
    client.delete_collection(c.name)
    print(f'Deleted: {c.name}')

# Also clean up the directory
if os.path.exists('/tmp/qdrant'):
    shutil.rmtree('/tmp/qdrant')
    print('Removed /tmp/qdrant')

# Also clean up ~/.hermes/qdrant if it exists
hermes_qdrant = os.path.expanduser('~/.hermes/qdrant')
if os.path.exists(hermes_qdrant):
    shutil.rmtree(hermes_qdrant)
    print(f'Removed {hermes_qdrant}')

print('Done')
