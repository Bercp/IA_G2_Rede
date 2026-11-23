"use strict";
const boardElement = document.getElementById("board");
const statusElement = document.getElementById("status");
const dialog = document.getElementById("promotion");
const colorButtons = [...document.querySelectorAll("[data-color]")];
const restart = document.getElementById("restart");
const names = {p: "peão", n: "cavalo", b: "bispo", r: "torre", q: "dama", k: "rei"};
const initial = {};
for (let file = 0; file < 8; file++) {
  const f = "abcdefgh"[file];
  initial[f + "1"] = "RNBQKBNR"[file]; initial[f + "2"] = "P";
  initial[f + "7"] = "p"; initial[f + "8"] = "rnbqkbnr"[file];
}
let humanColor = "white";
let state = null;
let selected = null;
let busy = false;
let dragged = false;

function pieceColor(piece) { return piece === piece.toUpperCase() ? "white" : "black"; }
function pieceImage(piece) { return `/pieces/${pieceColor(piece) === "white" ? "w" : "b"}${piece.toUpperCase()}.svg`; }
function canMove() { return !busy && state && !state.finished && state.turn === humanColor; }
function ownPiece(square) { const piece = state?.board[square]; return piece && pieceColor(piece) === humanColor; }

function render() {
  const pieces = state?.board || initial;
  const files = humanColor === "white" ? "abcdefgh" : "hgfedcba";
  const ranks = humanColor === "white" ? "87654321" : "12345678";
  const destinations = new Set((state?.legalMoves || []).filter(m => m.slice(0, 2) === selected).map(m => m.slice(2, 4)));
  const fragment = document.createDocumentFragment();
  for (const [row, rank] of [...ranks].entries()) {
    for (const [column, file] of [...files].entries()) {
      const square = file + rank;
      const piece = pieces[square];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "square " + (("abcdefgh".indexOf(file) + Number(rank)) % 2 ? "dark" : "light");
      button.dataset.square = square;
      button.setAttribute("aria-label", `${square}${piece ? ", " + names[piece.toLowerCase()] + " " + (pieceColor(piece) === "white" ? "branco" : "preto") : ", vazia"}`);
      button.setAttribute("aria-pressed", String(selected === square));
      if (selected === square) button.classList.add("selected");
      if (destinations.has(square)) button.classList.add("legal");
      if (state?.lastMove && [state.lastMove.slice(0, 2), state.lastMove.slice(2, 4)].includes(square)) button.classList.add("last");
      if (state?.checkSquare === square) button.classList.add("in-check");
      if (piece) {
        const img = document.createElement("img");
        img.src = pieceImage(piece); img.alt = ""; img.draggable = false;
        button.append(img); button.classList.add("occupied");
      }
      if (column === 0) { const label = document.createElement("span"); label.className = "coordinate rank"; label.textContent = rank; button.append(label); }
      if (row === 7) { const label = document.createElement("span"); label.className = "coordinate file"; label.textContent = file; button.append(label); }
      button.draggable = Boolean(canMove() && ownPiece(square));
      button.addEventListener("click", () => { if (!dragged) selectSquare(square); });
      button.addEventListener("dragstart", event => {
        if (!canMove() || !ownPiece(square)) { event.preventDefault(); return; }
        dragged = true;
        selected = square;
        event.dataTransfer.setData("text/plain", square);
        event.dataTransfer.effectAllowed = "move";
        // Keep the dragged DOM node alive until the browser completes the drag.
        for (const cell of boardElement.children) {
          cell.classList.toggle("selected", cell.dataset.square === square);
          cell.classList.toggle("legal", state.legalMoves.some(m => m.slice(0, 2) === square && m.slice(2, 4) === cell.dataset.square));
        }
      });
      button.addEventListener("dragover", event => { if (canMove()) { event.preventDefault(); event.dataTransfer.dropEffect = "move"; } });
      button.addEventListener("drop", event => {
        event.preventDefault();
        if (canMove()) attemptMove(event.dataTransfer.getData("text/plain"), square);
      });
      button.addEventListener("dragend", () => { setTimeout(() => { dragged = false; render(); }, 0); });
      fragment.append(button);
    }
  }
  boardElement.replaceChildren(fragment);
  boardElement.classList.toggle("busy", busy);
  boardElement.setAttribute("aria-busy", String(busy));
  for (const button of colorButtons) { button.setAttribute("aria-pressed", String(button.dataset.color === humanColor)); button.disabled = busy; }
  restart.disabled = busy;
}

function selectSquare(square) {
  if (!canMove()) return;
  if (selected === square) { selected = null; render(); return; }
  if (selected && attemptMove(selected, square)) return;
  selected = ownPiece(square) ? square : null;
  render();
}

function attemptMove(from, to) {
  if (!canMove()) return false;
  const candidates = state.legalMoves.filter(move => move.slice(0, 4) === from + to);
  if (!candidates.length) return false;
  if (candidates.length === 1) submitMove(candidates[0]);
  else {
    const options = document.getElementById("promotion-options");
    options.replaceChildren();
    for (const code of ["q", "r", "b", "n"]) {
      const move = candidates.find(m => m[4] === code);
      if (!move) continue;
      const button = document.createElement("button"); button.type = "button";
      const img = document.createElement("img"); img.src = pieceImage(humanColor === "white" ? code.toUpperCase() : code); img.alt = "";
      button.append(img, names[code]); button.setAttribute("aria-label", "Promover para " + names[code]);
      button.addEventListener("click", () => { dialog.close(); submitMove(move); });
      options.append(button);
    }
    dialog.showModal();
  }
  return true;
}

async function requestPosition(moves) {
  busy = true; selected = null;
  statusElement.classList.remove("error");
  statusElement.textContent = moves.length || humanColor === "black" ? "O modelo está pensando…" : "Preparando a partida…";
  render();
  try {
    const response = await fetch("/api/play", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({moves, color: humanColor})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Não foi possível completar a jogada.");
    state = result;
    statusElement.textContent = result.status;
  } catch (error) {
    statusElement.classList.add("error");
    statusElement.textContent = "Não foi possível jogar. Verifique se o servidor está aberto e tente novamente.";
    console.error(error);
  } finally {
    busy = false; dragged = false; render();
  }
}

function submitMove(move) { requestPosition([...state.moves, move]); }
function newGame(color = humanColor) {
  if (busy) return;
  humanColor = color; state = null; selected = null;
  requestPosition([]);
}
for (const button of colorButtons) button.addEventListener("click", () => newGame(button.dataset.color));
restart.addEventListener("click", () => newGame());
render();
newGame();
