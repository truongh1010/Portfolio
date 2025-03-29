import { useState } from "react";

export default function Player({
  inititalName,
  sympol,
  isActive,
  onChangeName,
}) {
  const [playerName, setPlayerName] = useState(inititalName);
  const [isEditing, setIsEditing] = useState(false);

  function handleEditClick() {
    setIsEditing((editing) => !editing);
    if (isEditing) {
      onChangeName(sympol, playerName);
    }
  }

  function handleChange(event) {
    console.log(event);
    setPlayerName(event.target.value);
  }

  let editedPlayerName = <span className="player-name">{playerName}</span>;
  // let buttonCaption = 'Edit'

  if (isEditing) {
    editedPlayerName = (
      <input type="text" required value={playerName} onChange={handleChange} />
    );
    // buttonCaption = 'Save'
  }

  return (
    <li className={isActive ? "active" : undefined}>
      <span className="player">
        {editedPlayerName}
        <span className="player-symplol">{sympol}</span>
      </span>
      <button onClick={handleEditClick}>{isEditing ? "Save" : "Edit"}</button>
    </li>
  );
}
