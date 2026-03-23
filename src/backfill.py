"""
Backfill embeddings for events stored before semantic retrieval existed.

Events ingested from Commit 60 onward are embedded as they are written. Rows
that predate it have a NULL embedding and are excluded from search, so the
index covers only recent reporting until this has run.

Safe to run repeatedly and safe to interrupt: it selects only rows that still
lack a vector, so a second run resumes rather than redoing work.
"""

import logging
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Small enough that an interrupted run loses little, large enough that the
# model is not reloaded per handful of rows.
DEFAULT_BATCH_SIZE = 100


def backfill(storage, embedder, batch_size: int = DEFAULT_BATCH_SIZE,
             max_batches: Optional[int] = None) -> Dict[str, int]:
    """Embed events that have no vector yet, a batch at a time"""
    totals = {'embedded': 0, 'failed': 0, 'batches': 0}

    while max_batches is None or totals['batches'] < max_batches:
        events = storage.get_events_without_embeddings(limit=batch_size)
        if not events:
            break

        try:
            vectors = embedder.embed_events(events)
            updated = storage.update_embeddings({
                event['event_id']: vector
                for event, vector in zip(events, vectors)
            })
            totals['embedded'] += updated
        except Exception as e:
            # Stop rather than spin: the same batch would be selected again on
            # the next pass, so continuing would loop on the failing rows.
            logger.error(f"Batch failed after {totals['embedded']} events: {e}")
            totals['failed'] += len(events)
            break

        totals['batches'] += 1
        logger.info(f"Embedded {totals['embedded']} events so far")

    logger.info(
        f"Backfill complete: {totals['embedded']} embedded, "
        f"{totals['failed']} failed, {totals['batches']} batches"
    )
    return totals


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Embed events that have no vector')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='stop after this many batches; default is all')
    args = parser.parse_args()

    from storage_manager import get_storage_manager
    from embeddings import get_embedder

    backfill(get_storage_manager(), get_embedder(),
             batch_size=args.batch_size, max_batches=args.max_batches)


if __name__ == '__main__':
    main()
