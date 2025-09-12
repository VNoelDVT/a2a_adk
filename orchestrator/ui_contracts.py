# orchestrator/ui_contracts.py
from typing import Any, Dict, List

__all__ = ["shortlist_view", "bu_health_view"]

def _fmt_eur(v: Any) -> str:
    try:
        return f"{int(v)}€"
    except Exception:
        return str(v) if v is not None else "n/a"

def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v):.0f}%"
    except Exception:
        return str(v) if v is not None else "n/a"

def _join(items: Any, sep: str = ", ") -> str:
    if not items:
        return "n/a"
    if isinstance(items, (list, tuple, set)):
        return sep.join(str(x) for x in items if x is not None)
    return str(items)

def shortlist_view(offer: Dict[str, Any], shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construit le rendu shortlist et le retourne sous la forme {'text': ...}.
    Le dernier tour de l'agent doit IMPRIMER ce champ `text` tel quel.
    """
    client = offer.get("client_id") or offer.get("client") or "Client"
    role = offer.get("role") or "Rôle"
    title = f"Top candidats pour {role} @ {client}\n"

    if not shortlist:
        return {"status": "success", "text": f"{title}\nAucun profil correspondant pour le moment."}

    lines = [title]
    for c in shortlist:
        cid    = c.get("consultant_id") or "C-XXX"
        grade  = c.get("grade") or "N/A"
        city   = c.get("city") or c.get("location") or "—"
        tjm    = _fmt_eur(c.get("tjm_min"))
        skills = _join(c.get("skills"))
        langs  = _join(c.get("languages"))
        yrs = c.get("years_experience")
        exp_line = f"Exp: {yrs} ans" if yrs is not None else ""
        av = c.get("availability") or {}
        load = av.get("load_pct")
        load_str = f"{load}%" if load is not None else "n/a"
        av_line = f"Disponibilité: {av.get('from', 'n/a')} / {load_str}" if av else ""

        lines.append(f"• {cid} — {grade} ({city}, TJM {tjm})")
        lines.append(f"  Skills: {skills}")
        lines.append(f"  Langues: {langs}")
        if exp_line: lines.append(f"  {exp_line}")
        if av_line:  lines.append(f"  {av_line}")
        lines.append("")

    return {"status": "success", "text": "\n".join(lines)}

def bu_health_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit le rendu BU health et le retourne sous la forme {'text': ...}.
    Le dernier tour de l'agent doit IMPRIMER ce champ `text` tel quel.
    """
    bu = report.get("bu_name") or report.get("name") or "BU"
    title = f"Rapport de santé — {bu}\n"
    lines = [title]

    kpis: Dict[str, Any] = report.get("kpis") or {}
    if kpis:
        lines.append("KPIs:")
        if "revenue_mtd" in kpis:  lines.append(f"  - CA MTD: {kpis['revenue_mtd']}")
        if "revenue_qtd" in kpis:  lines.append(f"  - CA QTD: {kpis['revenue_qtd']}")
        if "gross_margin" in kpis: lines.append(f"  - Marge: {_fmt_pct(kpis['gross_margin'])}")
        if "win_rate" in kpis:     lines.append(f"  - Taux de win: {_fmt_pct(kpis['win_rate'])}")
        lines.append("")

    utilization = report.get("utilization") or {}
    if utilization:
        lines.append("Utilisation:")
        if "overall" in utilization:  lines.append(f"  - Globale: {_fmt_pct(utilization['overall'])}")
        if "bench" in utilization:    lines.append(f"  - Bench: {_fmt_pct(utilization['bench'])}")
        if "billable" in utilization: lines.append(f"  - Billable: {_fmt_pct(utilization['billable'])}")
        lines.append("")

    pipeline = report.get("pipeline") or {}
    if pipeline:
        lines.append("Pipeline:")
        if "open_deals" in pipeline: lines.append(f"  - Opportunités ouvertes: {pipeline['open_deals']}")
        if "value" in pipeline:      lines.append(f"  - Valeur: {pipeline['value']}")
        if "next_30d" in pipeline:   lines.append(f"  - Closables 30j: {pipeline['next_30d']}")
        lines.append("")

    risks: List[Dict[str, Any]] = report.get("risks") or []
    if risks:
        lines.append("Risques:")
        for r in risks[:8]:
            name = r.get("name") or "Risque"
            sev  = r.get("severity")
            note = r.get("note") or r.get("description") or ""
            lines.append(f"  • {name} — {sev or 'n/a'} — {note}")
        lines.append("")

    actions: List[Dict[str, Any]] = report.get("actions") or []
    if actions:
        lines.append("Actions:")
        for a in actions[:8]:
            txt = a.get("title") or a.get("action") or "Action"
            owner = a.get("owner")
            due = a.get("due") or ""
            line = f"  • {txt}"
            if owner: line += f" — {owner}"
            if due:   line += f" — échéance {due}"
            lines.append(line)
        lines.append("")

    if len(lines) == 1:
        lines.append("Aucune donnée exploitable.")

    return {"status": "success", "text": "\n".join(lines)}
