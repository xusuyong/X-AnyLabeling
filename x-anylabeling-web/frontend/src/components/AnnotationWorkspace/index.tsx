import { Layout } from "antd";
import TopToolbar from "../toolbars/TopToolbar";
import AnnotationCanvas from "../canvas/AnnotationCanvas";
import FileBrowser from "../panels/FileBrowser";
import LabelList from "../panels/LabelList";
import { useProjectStore } from "../../stores/projectStore";

const { Sider, Content } = Layout;

export default function AnnotationWorkspace() {
  const { currentProject } = useProjectStore();

  if (!currentProject) return null;

  return (
    <Layout style={{ height: "100vh" }}>
      {/* Top toolbar */}
      <TopToolbar />

      <Layout>
        {/* Left: File browser */}
        <Sider
          width={200}
          style={{ background: "#fff", borderRight: "1px solid #e8e8e8" }}
        >
          <FileBrowser />
        </Sider>

        {/* Center: Canvas */}
        <Content style={{ display: "flex", flexDirection: "column" }}>
          <AnnotationCanvas />
        </Content>

        {/* Right: Label list */}
        <Sider
          width={220}
          style={{ background: "#fff", borderLeft: "1px solid #e8e8e8" }}
        >
          <div style={{ padding: "8px", borderBottom: "1px solid #e8e8e8" }}>
            <strong>Annotations</strong>
          </div>
          <LabelList />
        </Sider>
      </Layout>
    </Layout>
  );
}
