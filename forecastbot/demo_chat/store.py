"""Persistent demo sessions, isolated from experiment artifacts."""
from contextlib import contextmanager
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'sessions.sqlite3'
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS sessions
                (id TEXT PRIMARY KEY, title TEXT, config TEXT, updated TEXT);
                CREATE TABLE IF NOT EXISTS turns
                (id TEXT PRIMARY KEY, session_id TEXT, question TEXT, config TEXT,
                 status TEXT, result TEXT, parent_id TEXT, created TEXT);''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def auto_title(self,sid,title):
        with self.connect() as db:
            db.execute("UPDATE sessions SET title=? WHERE id=? AND title IN ('New session', 'Phiên mới')",(title[:100],sid))

    def create(self, title='New session'):
        sid = uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?)', (sid,title,'{}',now()))
        return sid

    def sessions(self):
        with self.connect() as db:
            sessions = [dict(r) for r in db.execute('SELECT * FROM sessions ORDER BY updated DESC')]
        for session in sessions:
            if session['title'] == 'Phiên mới':
                session['title'] = 'New session'
        return sessions

    def save_config(self,sid,config):
        with self.connect() as db:
            db.execute('UPDATE sessions SET config=? WHERE id=?',(json.dumps(config,ensure_ascii=False),sid))

    def rename(self,sid,title):
        with self.connect() as db:
            db.execute('UPDATE sessions SET title=?, updated=? WHERE id=?',(title.strip() or 'New session',now(),sid))

    def turns(self,sid):
        with self.connect() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM turns WHERE session_id=? ORDER BY created,id',(sid,))]
        for r in rows:
            r['config']=json.loads(r['config']);r['result']=json.loads(r['result']) if r['result'] else {}
        return rows

    def begin(self,sid,question,config,parent=None):
        self.auto_title(sid,' '.join(question.split())[:100])
        tid=uuid4().hex
        with self.connect() as db:
            if db.execute("SELECT 1 FROM turns WHERE session_id=? AND status='running'",(sid,)).fetchone():
                raise ValueError('This session already has a request in progress.')
            db.execute('INSERT INTO turns VALUES (?,?,?,?,?,?,?,?)',
                       (tid,sid,question,json.dumps(config,ensure_ascii=False),'running',None,parent,now()))
            db.execute('UPDATE sessions SET updated=? WHERE id=?',(now(),sid))
        return tid

    def finish(self,tid,result):
        status='needs_clarification' if result.get('forecast_request',{}).get('needs_clarification') else result.get('status','failed')
        with self.connect() as db:
            db.execute('UPDATE turns SET status=?, result=? WHERE id=?',
                       (status,json.dumps(result,ensure_ascii=False,default=str),tid))

    def interrupt(self,tid):
        with self.connect() as db:
            db.execute("UPDATE turns SET status='interrupted' WHERE id=? AND status='running'",(tid,))
