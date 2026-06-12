import { useEffect, useState } from "react";
import { Button, Card, Input, Modal, Space, Typography, message } from "antd";
import { PlusOutlined, FolderOpenOutlined } from "@ant-design/icons";
import { useProjectStore } from "../../stores/projectStore";
import type { Project } from "../../types/api";

const { Title, Text } = Typography;

export default function ProjectSelector({ onSelect }: { onSelect: (id: string) => void }) {
  const { projects, fetchProjects, createProject } = useProjectStore();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreate = async () => {
    if (!newName.trim()) {
      message.error("Project name is required");
      return;
    }
    const project = await createProject(newName.trim(), newDesc);
    setShowCreate(false);
    setNewName("");
    setNewDesc("");
    message.success(`Project "${project.name}" created`);
    onSelect(project.id);
  };

  return (
    <div style={{ maxWidth: 800, margin: "60px auto", padding: "0 24px" }}>
      <Title level={2} style={{ textAlign: "center", marginBottom: 32 }}>
        X-AnyLabeling Web
      </Title>

      <div style={{ marginBottom: 24, textAlign: "right" }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setShowCreate(true)}
        >
          New Project
        </Button>
      </div>

      {projects.length === 0 ? (
        <Card style={{ textAlign: "center", padding: 40 }}>
          <Text type="secondary" style={{ fontSize: 16 }}>
            No projects yet. Create one to get started.
          </Text>
        </Card>
      ) : (
        <Space direction="vertical" style={{ width: "100%" }} size={12}>
          {projects.map((p: Project) => (
            <Card
              key={p.id}
              hoverable
              onClick={() => onSelect(p.id)}
              style={{ cursor: "pointer" }}
            >
              <Card.Meta
                title={
                  <Space>
                    <FolderOpenOutlined />
                    {p.name}
                  </Space>
                }
                description={`${p.image_count} image(s) · ${p.description || "No description"}`}
              />
            </Card>
          ))}
        </Space>
      )}

      <Modal
        title="Create New Project"
        open={showCreate}
        onOk={handleCreate}
        onCancel={() => setShowCreate(false)}
        okText="Create"
      >
        <Space direction="vertical" style={{ width: "100%" }}>
          <Input
            placeholder="Project name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onPressEnter={handleCreate}
          />
          <Input.TextArea
            placeholder="Description (optional)"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            rows={2}
          />
        </Space>
      </Modal>
    </div>
  );
}
