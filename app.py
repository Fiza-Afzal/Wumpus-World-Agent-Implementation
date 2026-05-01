"""
Wumpus World — Python Flask Backend
All game logic (KB, resolution, agent AI) runs here.
Frontend just calls REST API endpoints.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# ─── In-memory game state ───────────────────────────────────────────────────
game_state = {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cell_key(r, c):
    return f"{r},{c}"

def pos_label(r, c):
    return f"({r},{c})"

def get_neighbors(r, c, rows, cols):
    neighbors = []
    if r > 0:          neighbors.append((r-1, c))
    if r < rows - 1:   neighbors.append((r+1, c))
    if c > 0:          neighbors.append((r, c-1))
    if c < cols - 1:   neighbors.append((r, c+1))
    return neighbors

def random_cell_excluding(excluded_keys, rows, cols):
    available = []
    for r in range(rows):
        for c in range(cols):
            if cell_key(r, c) not in excluded_keys:
                available.append((r, c))
    if not available:
        return None
    return random.choice(available)


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE (CNF Resolution)
# ══════════════════════════════════════════════════════════════════════════════

def negate_literal(lit):
    return lit[1:] if lit.startswith('!') else '!' + lit

def clause_exists(clauses, new_clause):
    sorted_new = ','.join(sorted(new_clause))
    for c in clauses:
        if ','.join(sorted(c)) == sorted_new:
            return True
    return False

def add_clause(kb, clause):
    if not clause_exists(kb, clause):
        kb.append(list(clause))

def resolve(c1, c2):
    results = []
    for lit in c1:
        neg_lit = negate_literal(lit)
        if neg_lit in c2:
            new_clause = [l for l in c1 if l != lit] + [l for l in c2 if l != neg_lit]
            deduped = list(dict.fromkeys(new_clause))  # remove dups, preserve order
            # Tautology check
            is_tautology = any(negate_literal(l) in deduped for l in deduped)
            if not is_tautology:
                results.append(deduped)
    return results

def ask_kb(kb, query, inference_steps):
    """Resolution refutation: add !query to KB copy, try to derive []."""
    negated = negate_literal(query)
    clauses = [list(c) for c in kb]
    clauses.append([negated])

    changed = True
    steps = 0
    log_msg = None

    while changed:
        changed = False
        steps += 1
        inference_steps[0] += 1

        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                resolvents = resolve(clauses[i], clauses[j])
                for res in resolvents:
                    if len(res) == 0:
                        log_msg = f"ASK: {query} — PROVED ({steps} steps)"
                        return True, log_msg, inference_steps[0]
                    if not clause_exists(clauses, res):
                        clauses.append(res)
                        changed = True
                if steps > 2000:
                    break
            if steps > 2000:
                break

    log_msg = f"ASK: {query} — cannot prove"
    return False, log_msg, inference_steps[0]


# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTION
# ══════════════════════════════════════════════════════════════════════════════

def get_percepts(gs, r, c):
    neighbors = get_neighbors(r, c, gs['rows'], gs['cols'])
    breeze = any(cell_key(nr, nc) in gs['pit_at'] for nr, nc in neighbors)
    wpos = gs['wumpus_pos']
    stench = gs['wumpus_alive'] and wpos and any(
        (nr == wpos[0] and nc == wpos[1]) for nr, nc in neighbors
    )
    gpos = gs['gold_pos']
    glitter = gpos and gpos[0] == r and gpos[1] == c

    percepts = []
    if breeze:   percepts.append('BREEZE')
    if stench:   percepts.append('STENCH')
    if glitter:  percepts.append('GLITTER')
    if not breeze and not stench:
        percepts.append('NONE')
    return percepts


# ══════════════════════════════════════════════════════════════════════════════
# KB UPDATE (TELL)
# ══════════════════════════════════════════════════════════════════════════════

def perceive_and_update_kb(gs, r, c):
    kb = gs['kb']
    percepts = get_percepts(gs, r, c)
    neighbors = get_neighbors(r, c, gs['rows'], gs['cols'])
    kb_log = gs['kb_log']

    # This cell is safe
    add_clause(kb, [f"SAFE_{r}_{c}"])
    kb_log.insert(0, f"TELL: SAFE_{r}_{c}")

    if 'BREEZE' in percepts:
        clause = [f"!B_{r}_{c}"] + [f"P_{nr}_{nc}" for nr, nc in neighbors]
        add_clause(kb, clause)
        add_clause(kb, [f"B_{r}_{c}"])
        nb_labels = ','.join(pos_label(nr, nc) for nr, nc in neighbors)
        kb_log.insert(0, f"TELL: B_{r}_{c} => pit in {{{nb_labels}}}")
        for nr, nc in neighbors:
            add_clause(kb, [f"!P_{nr}_{nc}", f"B_{r}_{c}"])
    else:
        add_clause(kb, [f"!B_{r}_{c}"])
        for nr, nc in neighbors:
            add_clause(kb, [f"!P_{nr}_{nc}"])
            gs['safe'][cell_key(nr, nc)] = True
            kb_log.insert(0, f"TELL: !P_{nr}_{nc} (no breeze at {pos_label(r,c)})")

    if 'STENCH' in percepts:
        clause = [f"!S_{r}_{c}"] + [f"W_{nr}_{nc}" for nr, nc in neighbors]
        add_clause(kb, clause)
        add_clause(kb, [f"S_{r}_{c}"])
        nb_labels = ','.join(pos_label(nr, nc) for nr, nc in neighbors)
        kb_log.insert(0, f"TELL: S_{r}_{c} => wumpus in {{{nb_labels}}}")
        for nr, nc in neighbors:
            add_clause(kb, [f"!W_{nr}_{nc}", f"S_{r}_{c}"])
    else:
        add_clause(kb, [f"!S_{r}_{c}"])
        if gs['wumpus_alive']:
            for nr, nc in neighbors:
                add_clause(kb, [f"!W_{nr}_{nc}"])
                kb_log.insert(0, f"TELL: !W_{nr}_{nc} (no stench at {pos_label(r,c)})")

    # Trim log
    gs['kb_log'] = kb_log[:40]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT PLANNING
# ══════════════════════════════════════════════════════════════════════════════

def plan_next_moves(gs):
    r, c = gs['agent_pos']
    neighbors = get_neighbors(r, c, gs['rows'], gs['cols'])
    inf_steps = [gs['inference_steps']]

    for nr, nc in neighbors:
        nkey = cell_key(nr, nc)
        if gs['visited'].get(nkey) or gs['done']:
            continue

        no_pit, msg1, _ = ask_kb(gs['kb'], f"!P_{nr}_{nc}", inf_steps)
        no_wumpus, msg2, _ = ask_kb(gs['kb'], f"!W_{nr}_{nc}", inf_steps)
        gs['kb_log'].insert(0, msg1)
        gs['kb_log'].insert(0, msg2)

        if no_pit and no_wumpus:
            gs['safe'][nkey] = True
            already_queued = any(cell_key(p[0], p[1]) == nkey for p in gs['to_visit'])
            if not already_queued:
                gs['to_visit'].append([nr, nc])
        else:
            if not no_pit:    gs['suspected_pit'][nkey] = True
            if not no_wumpus: gs['suspected_wumpus'][nkey] = True

    gs['inference_steps'] = inf_steps[0]
    gs['kb_log'] = gs['kb_log'][:40]


# ══════════════════════════════════════════════════════════════════════════════
# MOVE AGENT
# ══════════════════════════════════════════════════════════════════════════════

def move_agent_to(gs, r, c):
    key = cell_key(r, c)
    gs['agent_moves'] += 1

    # Pit?
    if key in gs['pit_at']:
        gs['confirmed_pit'][key] = True
        gs['done'] = True
        gs['agent_pos'] = [r, c]
        return {'result': 'game_over', 'reason': 'pit',
                'message': f'💀 Agent fell into a pit at {pos_label(r,c)}!'}

    # Wumpus?
    wp = gs['wumpus_pos']
    if wp and wp[0] == r and wp[1] == c and gs['wumpus_alive']:
        gs['done'] = True
        gs['agent_pos'] = [r, c]
        return {'result': 'game_over', 'reason': 'wumpus',
                'message': f'☠ Wumpus got the agent at {pos_label(r,c)}!'}

    # Gold?
    gp = gs['gold_pos']
    if gp and gp[0] == r and gp[1] == c:
        gs['agent_pos'] = [r, c]
        gs['visited'][key] = True
        gs['safe'][key] = True
        gs['done'] = True
        return {'result': 'victory',
                'message': f'🏆 Gold found at {pos_label(r,c)} in {gs["agent_moves"]} moves!'}

    # Normal move
    gs['agent_pos'] = [r, c]
    gs['visited'][key] = True
    gs['safe'][key] = True
    gs['suspected_pit'].pop(key, None)
    gs['suspected_wumpus'].pop(key, None)

    perceive_and_update_kb(gs, r, c)
    plan_next_moves(gs)

    return {'result': 'ok', 'message': f'Agent moved to {pos_label(r,c)}'}


# ══════════════════════════════════════════════════════════════════════════════
# SERIALISE STATE FOR FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

def serialise(gs):
    return {
        'rows': gs['rows'],
        'cols': gs['cols'],
        'agent_pos': gs['agent_pos'],
        'visited': gs['visited'],
        'safe': gs['safe'],
        'suspected_pit': gs['suspected_pit'],
        'suspected_wumpus': gs['suspected_wumpus'],
        'confirmed_pit': gs['confirmed_pit'],
        'wumpus_alive': gs['wumpus_alive'],
        'done': gs['done'],
        'inference_steps': gs['inference_steps'],
        'agent_moves': gs['agent_moves'],
        'kb_size': len(gs['kb']),
        'kb_log': gs['kb_log'][:20],
        'to_visit': gs['to_visit'],
        # percepts for current cell
        'current_percepts': get_percepts(gs, gs['agent_pos'][0], gs['agent_pos'][1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game_state
    data = request.get_json(force=True)
    rows     = max(3, min(8, int(data.get('rows', 4))))
    cols     = max(3, min(8, int(data.get('cols', 4))))
    num_pits = max(1, min(6, int(data.get('pits', 3))))

    start = (rows - 1, 0)
    excluded = {cell_key(*start)}

    # Place wumpus
    wp = random_cell_excluding(excluded, rows, cols)
    excluded.add(cell_key(*wp))

    # Place pits
    pit_at = {}
    for _ in range(num_pits):
        p = random_cell_excluding(excluded, rows, cols)
        if p:
            pit_at[cell_key(*p)] = True
            excluded.add(cell_key(*p))

    # Place gold (not at start)
    gold_excl = {cell_key(*start)}
    gp = random_cell_excluding(gold_excl, rows, cols)

    game_state = {
        'rows': rows, 'cols': cols,
        'pit_at': pit_at,
        'wumpus_pos': list(wp),
        'gold_pos': list(gp),
        'agent_pos': list(start),
        'visited': {cell_key(*start): True},
        'safe':    {cell_key(*start): True},
        'suspected_pit': {},
        'suspected_wumpus': {},
        'confirmed_pit': {},
        'wumpus_alive': True,
        'kb': [],
        'kb_log': [],
        'to_visit': [],
        'inference_steps': 0,
        'agent_moves': 0,
        'done': False,
    }

    perceive_and_update_kb(game_state, *start)
    plan_next_moves(game_state)

    return jsonify({'status': 'ok', 'state': serialise(game_state)})


@app.route('/api/step', methods=['POST'])
def step():
    gs = game_state
    if not gs or gs.get('done'):
        return jsonify({'status': 'error', 'message': 'Game not running'}), 400

    if not gs['to_visit']:
        plan_next_moves(gs)

    if not gs['to_visit']:
        gs['done'] = True
        return jsonify({
            'status': 'ok',
            'move_result': {'result': 'stuck', 'message': '🔍 No safe moves found. Exploration complete.'},
            'state': serialise(gs)
        })

    next_cell = gs['to_visit'].pop(0)
    result = move_agent_to(gs, next_cell[0], next_cell[1])

    return jsonify({'status': 'ok', 'move_result': result, 'state': serialise(gs)})


@app.route('/api/state', methods=['GET'])
def get_state():
    if not game_state:
        return jsonify({'status': 'error', 'message': 'No game'}), 404
    return jsonify({'status': 'ok', 'state': serialise(game_state)})


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
