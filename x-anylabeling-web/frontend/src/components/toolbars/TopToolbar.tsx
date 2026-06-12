import { Button, Space, Tooltip, Divider } from "antd";
import {
  SelectOutlined,
  BorderOutlined,
  DeleteOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { useAnnotationStore } from "../../stores/annotationStore";
import { useProjectStore } from "../../stores/projectStore";
import type { DrawMode } from "../../types/shape";

export default function TopToolbar() {
  const { drawMode, setDrawMode, removeShape, selectedShapeIndex } =
    useAnnotationStore();
  const { currentProject, currentImage } = useProjectStore();

  const handleDelete = () => {
    if (selectedShapeIndex >= 0) {
      removeShape(selectedShapeIndex);
      if (currentProject && currentImage) {
        const { saveAnnotation } = useAnnotationStore.getState();
        saveAnnotation(currentProject.id, currentImage.id);
      }
    }
  };

  const handleSave = () => {
    if (currentProject && currentImage) {
      const { saveAnnotation } = useAnnotationStore.getState();
      saveAnnotation(currentProject.id, currentImage.id);
    }
  };

  const handleZoom = (factor: number) => {
    // Zoom is handled by the canvas component
    // This is a placeholder for toolbar zoom buttons
    const canvas = document.querySelector("canvas");
    if (canvas) {
      canvas.dispatchEvent(
        new WheelEvent("wheel", { deltaY: factor > 1 ? -100 : 100, bubbles: true })
      );
    }
  };

  const drawModes: { mode: DrawMode; icon: React.ReactNode; label: string }[] = [
    { mode: "select", icon: <SelectOutlined />, label: "Select (V)" },
    { mode: "rectangle", icon: <BorderOutlined />, label: "Rectangle (R)" },
  ];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        padding: "4px 12px",
        background: "#fafafa",
        borderBottom: "1px solid #e8e8e8",
        gap: 4,
      }}
    >
      {/* Draw mode buttons */}
      <Space size={2}>
        {drawModes.map(({ mode, icon, label }) => (
          <Tooltip key={mode} title={label}>
            <Button
              type={drawMode === mode ? "primary" : "text"}
              icon={icon}
              size="small"
              onClick={() => setDrawMode(mode)}
            />
          </Tooltip>
        ))}
      </Space>

      <Divider type="vertical" />

      {/* Edit actions */}
      <Space size={2}>
        <Tooltip title="Delete (Del)">
          <Button
            icon={<DeleteOutlined />}
            size="small"
            danger
            disabled={selectedShapeIndex < 0}
            onClick={handleDelete}
          />
        </Tooltip>
      </Space>

      <Divider type="vertical" />

      {/* Zoom */}
      <Space size={2}>
        <Tooltip title="Zoom In">
          <Button icon={<ZoomInOutlined />} size="small" onClick={() => handleZoom(1.2)} />
        </Tooltip>
        <Tooltip title="Zoom Out">
          <Button icon={<ZoomOutOutlined />} size="small" onClick={() => handleZoom(0.8)} />
        </Tooltip>
      </Space>

      <div style={{ flex: 1 }} />

      {/* Save */}
      <Tooltip title="Save (Ctrl+S)">
        <Button
          icon={<SaveOutlined />}
          size="small"
          type="primary"
          onClick={handleSave}
        >
          Save
        </Button>
      </Tooltip>
    </div>
  );
}
