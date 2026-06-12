import { List, Typography, Tag, Space } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { useAnnotationStore } from "../../stores/annotationStore";
import type { Shape } from "../../types/shape";

const { Text } = Typography;

const SHAPE_COLORS: Record<string, string> = {
  rectangle: "blue",
  polygon: "green",
  point: "orange",
  line: "purple",
  circle: "cyan",
  rotation: "magenta",
  quadrilateral: "geekblue",
  linestrip: "volcano",
  cuboid: "gold",
};

export default function LabelList() {
  const { annotation, selectedShapeIndex, selectShape, removeShape } =
    useAnnotationStore();

  if (!annotation) return null;

  const shapes = annotation.shapes;

  return (
    <List
      size="small"
      dataSource={shapes}
      renderItem={(shape: Shape, index: number) => (
        <List.Item
          onClick={() => selectShape(index)}
          style={{
            cursor: "pointer",
            background: selectedShapeIndex === index ? "#e6f7ff" : undefined,
            padding: "2px 8px",
          }}
          actions={[
            <DeleteOutlined
              key="del"
              style={{ color: "#ff4d4f" }}
              onClick={(e) => {
                e.stopPropagation();
                removeShape(index);
              }}
            />,
          ]}
        >
          <Space size={4}>
            <Tag color={SHAPE_COLORS[shape.shape_type] || "default"} style={{ fontSize: 10, margin: 0 }}>
              {shape.shape_type}
            </Tag>
            <Text ellipsis style={{ maxWidth: 100 }} title={shape.label}>
              {shape.label}
            </Text>
            {shape.score != null && (
              <Text type="secondary" style={{ fontSize: 10 }}>
                {shape.score.toFixed(2)}
              </Text>
            )}
          </Space>
        </List.Item>
      )}
    />
  );
}
