import { Upload, List, Typography, Button } from "antd";
import { InboxOutlined, DeleteOutlined } from "@ant-design/icons";
import { useProjectStore } from "../../stores/projectStore";

const { Dragger } = Upload;
const { Text } = Typography;

export default function FileBrowser() {
  const {
    currentProject,
    images,
    currentImage,
    selectImage,
    uploadImages,
    deleteImage,
  } = useProjectStore();

  if (!currentProject) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <Dragger
        multiple
        accept="image/*"
        showUploadList={false}
        customRequest={({ file, onSuccess }) => {
          uploadImages([file as File]).then(() => onSuccess?.("ok"));
        }}
        style={{ padding: "8px 12px", marginBottom: 8 }}
      >
        <p className="ant-upload-drag-icon" style={{ margin: "4px 0" }}>
          <InboxOutlined style={{ fontSize: 20 }} />
        </p>
        <p style={{ margin: 0, fontSize: 12 }}>Drop images or click to upload</p>
      </Dragger>

      <div style={{ flex: 1, overflow: "auto" }}>
        <List
          size="small"
          dataSource={images}
          renderItem={(img) => (
            <List.Item
              onClick={() => selectImage(img.id)}
              style={{
                cursor: "pointer",
                background: currentImage?.id === img.id ? "#e6f7ff" : undefined,
                padding: "4px 8px",
              }}
              actions={[
                <Button
                  key="del"
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteImage(img.id);
                  }}
                />,
              ]}
            >
              <Text ellipsis style={{ maxWidth: 140 }} title={img.filename}>
                {img.filename}
              </Text>
            </List.Item>
          )}
        />
      </div>
    </div>
  );
}
