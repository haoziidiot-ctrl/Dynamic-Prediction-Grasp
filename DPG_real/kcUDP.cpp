#include"kcUDP.h"
//中文乱码问题
#ifdef WIN32  
#pragma execution_character_set("utf-8")  
#endif
//3D密钥
//de-c9-9d-62-0a-8a-f9-4a-a4-28-f4-0e-fc-46-11-36
//2D密钥
//31-04-25-49-2b-32-af-48-8f-f1-82-ee-8b-93-5e-68
kcUDP::kcUDP(QString& kc_key, QObject* parent)
{
	kc_key = "31-04-25-49-2b-32-af-48-8f-f1-82-ee-8b-93-5e-68";
	devicekey = hexToQByteArray(kc_key);
	//qDebug() << devicekey.toHex(' '); // 输出原始HEX格式

	//QString input = "NaviMode";
	//QByteArray result = convertTo16Array(input);
	//qDebug() << result.toHex(' '); // 输出原始HEX格式

	my_thread = new QThread();
	//绑定端口号
	udpSocket = new QUdpSocket(this);
	udpSocket->bind(8888);//可不绑定
	this->moveToThread(my_thread);
	udpSocket->moveToThread(my_thread);

	m_statusTimer = new QTimer();
	m_statusTimer->start(100);
	m_statusTimer->moveToThread(my_thread);

	my_thread->start();
	connect(my_thread, &QThread::finished, my_thread, &QObject::deleteLater);
	connect(my_thread, &QThread::finished, udpSocket, &QObject::deleteLater);
	//当对方发送数据过来。自动触发readyRead()
	//connect(udpSocket, &QUdpSocket::readyRead, this, &kcUDP::dealMsg);//存到本地 之后再写
	connect(udpSocket, &QUdpSocket::readyRead, this, &kcUDP::handle_data, Qt::DirectConnection);

	connect(m_statusTimer, SIGNAL(timeout()), this, SLOT(Inint()), Qt::QueuedConnection);

};
kcUDP::~kcUDP()
{
	udpSocket->disconnectFromHost();
	udpSocket->close();
	my_thread->quit();
	my_thread->wait();
}
void kcUDP::dealMsg()
{
	//读取对方发送的内容
	char buf[1024] = { 0 };           //内容
	QHostAddress cliAddr;           //对方地址
	quint16 port;                   //对方端口
	qint64 len = udpSocket->readDatagram(buf, sizeof(buf), &cliAddr, &port);
	QTime cur_time = QTime::currentTime();
	QString time_info = cur_time.toString("hh:mm:ss");
	if (len > 0)
	{
		//获取到文本格式化  [192.168.1.1 : 8888]aaaa
		QString str = QString("[%1:%2] %3 [%4]")
			.arg(cliAddr.toString())
			.arg(port)
			.arg(buf)
			.arg(time_info);
		qDebug() << str;
	}
}
//返回应答
void kcUDP::handle_data()
{
	QHostAddress cliAddr;           //对方地址
	quint16 port;                   //对方端口
	QByteArray datagram;
	datagram.resize(udpSocket->pendingDatagramSize());
	udpSocket->readDatagram(datagram.data(), datagram.size(), &cliAddr, &port);
	QString hexStr = datagram.toHex(' ').toUpper(); // 转HEX格式
	RobotData data = parseResponse(datagram);
	formatData(data);

	qDebug() << hexStr;
}
//void kcUDP::Inint()
//{
//	// 轮流发送状态查询命令
//	if (!m_sendLocationNext) {
//		CheckRunningStatus();
//	}
//	else {
//		GetLocation();
//	}
//	m_sendLocationNext = !m_sendLocationNext;
//	// 状态机处理逻辑
//	bool processed = false;
//	if (!processed && !m_navigating) {
//		processed = true;
//		// 步骤2: 判断SwitchModel 和地图状态
//		if (m_switchModel == 0 && m_mapLoad == 0) {
//			// 步骤3: 判断m_navigationStatus状态
//			qDebug() << "地图载入成功";
//			switch (m_navigationStatus) {//机器人定位状态
//			case 1: //成功
//				//Confirmlocation();
//				if (confidenceUDP > 0.9)
//				{
//					if (!initUDP) {
//						Confirmlocation();
//					}
//					else {
//						if (!SwitchtoA)
//						{
//							SwitchTOautomatic(); //切换自动
//						}
//					}
//				}
//				else {
//					ExecuteManually();
//				}
//				break;
//			case 0: // 失败
//				ExecuteManually();//手动定位
//				break;
//			case 2: // 
//				//定位中等待
//				break;
//			case 3: // 定位成功
//				if (confidenceUDP > 0.9)
//				{
//					Confirmlocation();
//				}
//				break;
//			default:
//				break;
//			}
//		}
//		else if (m_switchModel == 1 && m_mapLoad == 0)
//		{
//			// 步骤4: 判断Navigation状态
//			switch (m_taskStatus) {
//			case 0: // 无导航点任务
//				NavigationControl();//开始导航
//				m_taskStart = true;
//				break;
//			case 1: // 等待
//			case 2: // 正在前往导航点
//			case 3: // 暂停
//			case 4: // 完成
//			case 5: // 失败
//			case 6: // 退出
//			case 7: // 等待开/关门
//				// 步骤10: 不进行任何操作，小车正在运行中
//				qDebug() << "小车正在运行中，不执行额外操作";
//				break;
//			default:
//				break;
//			}
//		}
//		else if (m_mapLoad == 1) { qDebug() << "载入失败"; }
//		else if (m_mapLoad == 2) { qDebug() << "未载入地图 "; }
//		else if (m_mapLoad == 3) { qDebug() << "正在载入地图 "; }
//	}
//	if (!processed && m_navigating)
//	{
//		processed = true;
//		// 步骤4: 判断Navigation状态
//		switch (m_taskStatus) {
//		case 0:
//		case 1: // 等待
//		case 2: // 正在前往导航点
//		case 3: // 暂停
//		case 4: // 完成
//			if (m_taskStart)
//			{
//				m_taskEnd = true;
//				m_taskStart = false;
//				NavigationControlBack();
//			}
//			break;
//		case 5: // 失败
//		case 6: // 退出
//		case 7: // 等待开/关门
//			// 步骤10: 不进行任何操作，小车正在运行中
//			qDebug() << "小车正在运行中，不执行额外操作";
//			break;
//		default:
//			break;
//		}
//	}
//}


//优化后
void kcUDP::Inint() {

	static ControlPhase controlPhase = PHASE_CHECK_STATUS; // 控制阶段状态

	// 轮流执行三种操作：状态查询、位置查询、控制命令
	if (!m_sendLocationNext) {
		// 第一个100ms：发送状态查询命令
		CheckRunningStatus();  // 包含所有状态码的查询
	}
	else {
		// 第二个100ms：执行位置查询或控制逻辑
		if (!m_controlMode) {
			// 位置查询模式
			GetLocation();
		}
		else {
			// 控制指令模式：执行状态机处理逻辑
			ExecuteControlLogic();
		}

		// 在位置查询和控制模式间切换
		m_controlMode = !m_controlMode;
	}

	m_sendLocationNext = !m_sendLocationNext;
}
void kcUDP::ExecuteControlLogic()
{
	bool processed = false;

	if (!processed && !m_navigating) {
		processed = true;

		// 步骤2: 判断SwitchModel 和地图状态
		if (m_switchModel == 0 && m_mapLoad == 0) {
			// 步骤3: 判断m_navigationStatus状态
			qDebug() << "地图载入成功，执行控制逻辑";

			switch (m_navigationStatus) {
			case 1: // 成功
				if (confidenceUDP > 0.95) {
					if (!initUDP) {
						Confirmlocation();
					}
					else {
						if (!SwitchtoA) {
							SwitchTOautomatic(); // 切换自动
						}
					}
				}
				else {
					ExecuteManually();
				}
				break;

			case 0: // 失败
				ExecuteManually(); // 手动定位
				break;

			case 2: // 定位中等待
				// 定位中等待，不发送命令
				break;

			case 3: // 定位成功
				if (confidenceUDP > 0.95) {
					Confirmlocation();
				}
				break;

			default:
				break;
			}
		}
		else if (m_switchModel == 1 && m_mapLoad == 0) {
			// 步骤4: 判断Navigation状态
			switch (m_taskStatus) {
			case 0: // 无导航点任务
				NavigationControl(); // 开始导航
				m_taskStart = true;
				break;

			case 1: // 等待
			case 2: // 正在前往导航点
			case 3: // 暂停
			case 4: // 完成
			case 5: // 失败
			case 6: // 退出
			case 7: // 等待开/关门
				qDebug() << "小车正在运行中，不执行额外操作";
				break;

			default:
				break;
			}
		}
		else if (m_mapLoad == 1) {
			qDebug() << "载入失败";
		}
		else if (m_mapLoad == 2) {
			qDebug() << "未载入地图";
		}
		else if (m_mapLoad == 3) {
			qDebug() << "正在载入地图";
		}
	}

	if (!processed && m_navigating) {
		// 步骤4: 判断Navigation状态
		switch (m_taskStatus) {
		case 0:
		case 1: // 等待
		case 2: // 正在前往导航点
		case 3: // 暂停
		case 4: // 完成
			if (m_taskStart && !m_taskEnd) {//开始未返航
				m_taskEnd = true;
				m_taskStart = false;
				m_taskReCall = false;
				NavigationControlBack();
			}
			if (m_taskReCall && m_taskEnd)//返航后重开
			{
				m_taskEnd = false;
				m_taskStart = true;
				m_taskReCall = false;
				NavigationControl();
			}
			break;

		case 5: // 失败
		case 6: // 退出
		case 7: // 等待开/关门
			qDebug() << "小车正在运行中，不执行额外操作";
			break;

		default:
			break;
		}
	}
}

// 定义字节序转换辅助函数
double readDouble(const QByteArray& data, int offset) {
	quint64 value = 0;
	for (int i = 0; i < 8; ++i) {
		value |= static_cast<quint64>(static_cast<quint8>(data[offset + i])) << (i * 8);
	}
	return *reinterpret_cast<double*>(&value);
}

double readDoubleBigEndian(const QByteArray& data, int offset) {
	quint64 value = 0;
	for (int i = 0; i < 8; ++i) {
		value |= static_cast<quint64>(static_cast<quint8>(data[offset + i])) << ((7 - i) * 8);
	}
	return *reinterpret_cast<double*>(&value);
}

float readFloat(const QByteArray& data, int offset) {
	quint32 value = 0;
	for (int i = 0; i < 4; ++i) {
		value |= static_cast<quint32>(static_cast<quint8>(data[offset + i])) << (i * 8);
	}
	return *reinterpret_cast<float*>(&value);
}

quint16 readUInt16(const QByteArray& data, int offset) {
	return static_cast<quint16>(static_cast<quint8>(data[offset])) |
		(static_cast<quint16>(static_cast<quint8>(data[offset + 1])) << 8);
}

quint16 readUInt16BigEndian(const QByteArray& data, int offset) {
	return static_cast<quint16>(static_cast<quint8>(data[offset])) << 8 |
		static_cast<quint16>(static_cast<quint8>(data[offset + 1]));
}

quint32 readUInt32(const QByteArray& data, int offset) {
	return static_cast<quint32>(static_cast<quint8>(data[offset])) |
		(static_cast<quint32>(static_cast<quint8>(data[offset + 1])) << 8) |
		(static_cast<quint32>(static_cast<quint8>(data[offset + 2])) << 16) |
		(static_cast<quint32>(static_cast<quint8>(data[offset + 3])) << 24);
}

quint64 readUInt64(const QByteArray& data, int offset) {
	quint64 value = 0;
	for (int i = 0; i < 8; ++i) {
		value |= static_cast<quint64>(static_cast<quint8>(data[offset + i])) << (i * 8);
	}
	return value;
}

quint8 readUInt8(const QByteArray& data, int offset) {
	return static_cast<quint8>(data[offset]);
}

RobotData kcUDP::parseResponse(const QByteArray& data) {
	RobotData result;

	// 提取各字段
	result.robotKey = data.mid(0, 16);
	result.protocolVersion = readUInt8(data, 16);
	result.messageType = readUInt8(data, 17);
	result.communicationSequence = readUInt16(data, 18);
	result.serviceCode = readUInt8(data, 20);
	result.commandCode = readUInt8(data, 21);
	result.executeCode = readUInt8(data, 22);
	result.reserved1 = readUInt8(data, 23);
	result.messageLength = readUInt16(data, 24);//实测小端(应该是跟工控机配置有关)

	if (result.commandCode == 0x16 && result.executeCode == 0)//开始导航已执行
	{
		m_navigating = true;
	}
	if (result.commandCode == 0x14 && result.executeCode == 0)//手动定位操作已执行
	{
		m_isExecutingManual = true;
	}
	if (result.commandCode == 0x17 && result.executeCode == 0)//确认位置已执行
	{
		initUDP = true;
	}
	if (result.commandCode == 0x11 && result.executeCode == 0)//手自动切换完成
	{
		if (SwitchtoM) {
			SwitchtoM = false;
			SwitchtoA = true;
		}
		if (SwitchtoA) {
			SwitchtoM = true;
			SwitchtoA = false;
		}
	}
	if (result.messageLength == 0) {
		return result;
	}
	result.reserved2 = readUInt16(data, 26);

	if (result.commandCode == 0x17) {//查询机器人运行状态
		result.bodyTemperature = readDouble(data, 28);
		result.xCoordinate = readDouble(data, 36);
		result.yCoordinate = readDouble(data, 44);
		result.orientationAngle = readDouble(data, 52) * 180 / 3.1415926;//这儿与说明书不对
		result.batteryLevel = readDouble(data, 60);
		result.isBlocked = readUInt8(data, 68);
		result.isCharging = readUInt8(data, 69);
		result.runMode = readUInt8(data, 70);//运行模式 0手动 1自动
		m_switchModel = result.runMode;
		result.mapLoadStatus = readUInt8(data, 71);//地图是否载入成功
		m_mapLoad = result.mapLoadStatus;
		result.currentTargetId = readUInt32(data, 72);
		result.forwardSpeed = readDouble(data, 76);
		result.turnSpeed = readDouble(data, 84);
		result.batteryVoltage = readDouble(data, 92);
		result.current = readDouble(data, 100);
		result.taskStatus = readUInt8(data, 108);//导航任务状态
		m_taskStatus = result.taskStatus;
		result.positioningMode = readUInt8(data, 109);
		result.mapVersion = readUInt16(data, 110);
		result.reserved3 = readUInt32(data, 112);
		result.totalMileage = readDouble(data, 116);
		result.currentRunTime = readDouble(data, 124);
		result.totalRunTime = readDouble(data, 132);
		result.positioningStatus = readUInt8(data, 140);//定位状态 0 成功 1失败 2定位中 3定位成功
		m_navigationStatus = result.positioningStatus;
		result.trafficControl = readUInt8(data, 141);
		result.reserved4 = readUInt16(data, 142);
		result.mapCount = readUInt32(data, 144);
		QByteArray mapNameBytes = data.mid(148, 44);
		int nullIndex = mapNameBytes.indexOf('\0');
		if (nullIndex != -1) {
			mapNameBytes = mapNameBytes.left(nullIndex);
		}
		result.currentMapName = QString::fromUtf8(mapNameBytes);
		result.confidence = readDouble(data, 192);

		return result;
	}

	if (result.commandCode == 0x15) {// 获取机器人当前位置
		result.x = readDouble(data, 28);
		result.y = readDouble(data, 36);
		result.angle = readDouble(data, 44) * 180 / 3.1415926;
		result.confidence = readDouble(data, 52); confidenceUDP = result.confidence;
		result.totalLasers = readUInt16(data, 60);
		result.validLasers = readUInt16(data, 62);
		result.matchedLasers = readUInt16(data, 64);
		result.matchedColumns = readUInt16(data, 66);
		result.pathPointId = readUInt16(data, 68);
		result.pathId = readUInt16(data, 70);
		result.firstHighAreaId = readUInt16(data, 72);
		result.obstacleGroupNumber = readUInt16(data, 74);
		result.odomX = readDouble(data, 76);
		result.odomY = readDouble(data, 84);
		result.odomAngle = readDouble(data, 92);
		result.currentHighAreaId = readUInt64(data, 100);
		return result;
	}

	if (result.commandCode == 0xAF) {
		int i = 28;
		result.abnormal_size = readUInt8(data, i); i = i + 1;
		result.action_size = readUInt8(data, i); i = i + 1;
		result.Info_size = readUInt8(data, i); i = i + 2;
		//31是预留码
		//LocationStatusInfo[] 位置状态信息结构体
		result.GlobalX = readFloat(data, i); i = i + 4;
		result.GlobalY = readFloat(data, i); i = i + 4;
		result.GlobalA = readFloat(data, i) * 180 / 3.1415926; i = i + 4;
		//result.GlobalA = readUInt32(data, i); i = i + 4;
		result.endID = readUInt32(data, i); i = i + 4;
		result.endParagraph = readUInt32(data, i); i = i + 4;
		result.Dotserianumber = readUInt32(data, i); i = i + 4;
		result.confidenceAF = readUInt8(data, i); i = i + 1;
		result.positioningStatus = readUInt8(data, i); i = i + 7;
		//51-56是预留码
		//运行状态 RunningStatusInfo 参数
		result.LinearVelocity = readFloat(data, i); i = i + 4;
		result.Acceleration = readFloat(data, i); i = i + 4;
		result.AngularVelocity = readFloat(data, i); i = i + 4;
		result.WorkMode = readUInt8(data, i); i = i + 1;
		result.AGVStatus = readUInt8(data, i); i = i + 1;
		result.AbilitySettings = readUInt8(data, i); i = i + 6;
		//预留5个
		//任务状态 TaskStatusInfo 参数
		result.OrderID = readUInt32(data, i); i = i + 4;
		result.OrderKEY = readUInt32(data, i); i = i + 4;
		result.PointStateSequence = readUInt8(data, i); i = i + 1;
		result.PathStateSequence = readUInt8(data, i); i = i + 3;
		//预留2个
		if (result.PointStateSequence == 0) {
			//点状态 PointStateSequence 参数
		}
		if (result.PathStateSequence == 0) {
			//线状态 PathStateSequence 参数
		}
		//电池状态 BatteryStatusInfo 参数
		result.Batterylevel = readFloat(data, i); i = i + 4;
		result.VoltageAF = readFloat(data, i); i = i + 4;
		result.ElectricCurrent = readFloat(data, i); i = i + 4;
		result.ChargingStatus = readUInt8(data, i); i = i + 8;
		//预留7个
		if (result.abnormal_size != 0) { ; }//异常信息段
		if (result.action_size != 0) { ; }//异常信息段
		if (result.Info_size != 0) { ; }//异常信息段
		return result;
	}

	return result;
}

QString kcUDP::formatData(const RobotData& data) {
	QString info;
	info += QString("机器人密钥: %1\n").arg(QString(data.robotKey.toHex(' ')).toUpper());
	info += QString("协议版本号: %1\n").arg(data.protocolVersion);
	info += QString("报文类型: %1\n").arg(data.messageType);
	info += QString("通信序列: %1\n").arg(data.communicationSequence);
	info += QString("服务码: %1\n").arg(data.serviceCode);
	info += QString("命令码: %1\n").arg(data.commandCode);
	info += QString("执行码: %1\n").arg(data.executeCode);
	info += QString("预留码1: %1\n").arg(data.reserved1);
	info += QString("报文区域长度: %1\n").arg(data.messageLength);
	info += QString("预留码2: %1\n").arg(data.reserved2);

	qDebug() << QString("机器人密钥: %1").arg(QString(data.robotKey.toHex(' ')).toUpper());
	qDebug() << QString("协议版本号: %1").arg(data.protocolVersion);
	qDebug() << QString("报文类型: %1").arg(data.messageType);
	qDebug() << QString("通信序列: %1").arg(data.communicationSequence);
	qDebug() << QString("服务码: %1").arg(data.serviceCode);
	qDebug() << QString("命令码: %1").arg(data.commandCode);
	qDebug() << QString("执行码: %1").arg(data.executeCode);
	qDebug() << QString("预留码1: %1").arg(data.reserved1);
	qDebug() << QString("报文区域长度: %1").arg(data.messageLength);

	if (data.messageLength == 0) {
		return info;
	}

	if (data.commandCode == 0x15) {
		info += QString("机器人 X 坐标: %1\n").arg(data.x, 0, 'f', 2);
		info += QString("机器人 Y 坐标: %1\n").arg(data.y, 0, 'f', 2);
		info += QString("机器人朝向角度: %1\n").arg(data.angle, 0, 'f', 2);
		info += QString("定位置信度: %1\n").arg(data.confidence, 0, 'f', 2);
		info += QString("总激光数量: %1\n").arg(data.totalLasers);
		info += QString("有效激光数量: %1\n").arg(data.validLasers);
		info += QString("匹配到的有效激光数量: %1\n").arg(data.matchedLasers);
		info += QString("匹配到的反光柱数量: %1\n").arg(data.matchedColumns);
		info += QString("机器人所在路径点 ID: %1\n").arg(data.pathPointId);
		info += QString("机器人所在路径 ID: %1\n").arg(data.pathId);
		info += QString("机器人所在第一个高级区域的ID: %1\n").arg(data.firstHighAreaId);
		info += QString("机器人所在高级区域的避障分组编号: %1\n").arg(data.obstacleGroupNumber);
		info += QString("机器人里程计 X 坐标: %1\n").arg(data.odomX, 0, 'f', 2);
		info += QString("机器人里程计 Y 坐标: %1\n").arg(data.odomY, 0, 'f', 2);
		info += QString("机器人里程计角度: %1\n").arg(data.odomAngle, 0, 'f', 2);
		info += QString("当前所在高级区域 ID: %1\n").arg(data.currentHighAreaId);


		qDebug() << QString("预留码2: %1").arg(data.reserved2);
		qDebug() << QString("机器人 X 坐标: %1").arg(data.x, 0, 'f', 2);
		qDebug() << QString("机器人 Y 坐标: %1").arg(data.y, 0, 'f', 2);
		qDebug() << QString("机器人朝向角度: %1").arg(data.angle, 0, 'f', 2);
		qDebug() << QString("定位置信度: %1").arg(data.confidence, 0, 'f', 2);
		qDebug() << QString("总激光数量: %1").arg(data.totalLasers);
		qDebug() << QString("有效激光数量: %1").arg(data.validLasers);
		qDebug() << QString("匹配到的有效激光数量: %1").arg(data.matchedLasers);
		qDebug() << QString("匹配到的反光柱数量: %1").arg(data.matchedColumns);
		qDebug() << QString("机器人所在路径点 ID: %1").arg(data.pathPointId);
		qDebug() << QString("机器人所在路径 ID: %1").arg(data.pathId);
		qDebug() << QString("机器人所在第一个高级区域的ID: %1").arg(data.firstHighAreaId);
		qDebug() << QString("机器人所在高级区域的避障分组编号: %1").arg(data.obstacleGroupNumber);
		qDebug() << QString("机器人里程计 X 坐标: %1").arg(data.odomX, 0, 'f', 2);
		qDebug() << QString("机器人里程计 Y 坐标: %1").arg(data.odomY, 0, 'f', 2);
		qDebug() << QString("机器人里程计角度: %1").arg(data.odomAngle, 0, 'f', 2);
		qDebug() << QString("当前所在高级区域 ID: %1").arg(data.currentHighAreaId);
		return info;
	}

	if (data.commandCode == 0x17) {
		info += QString("本体温度: %1°C\n").arg(data.bodyTemperature, 0, 'f', 2);
		info += QString("位置的 X 坐标: %1m\n").arg(data.xCoordinate, 0, 'f', 3);
		info += QString("位置的 Y 坐标: %1m\n").arg(data.yCoordinate, 0, 'f', 3);
		info += QString("位置的朝向角度: %1°\n").arg(data.orientationAngle, 0, 'f', 3);
		info += QString("电池电量: %1%\n").arg(data.batteryLevel * 100, 0, 'f', 2);
		info += QString("是否被阻挡: %1\n").arg(data.isBlocked ? "是" : "否");
		info += QString("是否在充电: %1\n").arg(data.isCharging ? "是" : "否");
		info += QString("运行模式: %1\n").arg(data.runMode == 0 ? "手动" : "自动");
		info += QString("地图载入状态: %1\n").arg(data.mapLoadStatus == 0 ? "成功" : (data.mapLoadStatus == 1 ? "失败" : (data.mapLoadStatus == 2 ? "未载入" : "载入中")));
		info += QString("当前的目标点 id: %1\n").arg(data.currentTargetId);
		info += QString("前进速度: %1m/s\n").arg(data.forwardSpeed, 0, 'f', 3);
		info += QString("转弯速度: %1rad/s\n").arg(data.turnSpeed, 0, 'f', 3);
		info += QString("电池电压: %1V\n").arg(data.batteryVoltage, 0, 'f', 2);
		info += QString("电流: %1A\n").arg(data.current, 0, 'f', 2);
		info += QString("当前任务状态: %1\n").arg(data.taskStatus == 0 ? "无任务" : (data.taskStatus == 1 ? "等待" : (data.taskStatus == 2 ? "前往导航点" : (data.taskStatus == 3 ? "暂停" : (data.taskStatus == 4 ? "完成" : (data.taskStatus == 5 ? "失败" : (data.taskStatus == 6 ? "退出" : "等待开/关门")))))));
		info += QString("当前定位方式: %1\n").arg(data.positioningMode == 0 ? "未知" : (data.positioningMode == 1 ? "二维码" : (data.positioningMode == 2 ? "磁定位" : (data.positioningMode == 3 ? "激光无反" : (data.positioningMode == 4 ? "激光有反" : (data.positioningMode == 5 ? "高反码" : "GPS"))))));
		info += QString("地图版本号: %1\n").arg(data.mapVersion);
		info += QString("保留字段: %1\n").arg(data.reserved3);
		info += QString("累计行驶里程: %1m\n").arg(data.totalMileage, 0, 'f', 2);
		info += QString("本次运行时间: %1ms\n").arg(data.currentRunTime, 0, 'f', 2);
		info += QString("累计运行时间: %1ms\n").arg(data.totalRunTime, 0, 'f', 2);
		info += QString("机器人定位状态: %1\n").arg(data.positioningStatus == 0 ? "失败" : (data.positioningStatus == 1 ? "成功" : (data.positioningStatus == 2 ? "定位中" : "定位完成")));
		info += QString("交通管制: %1\n").arg(data.trafficControl ? "是" : "否");
		info += QString("保留字段: %1\n").arg(data.reserved4);
		info += QString("地图数量: %1\n").arg(data.mapCount);
		info += QString("当前地图名称: %1\n").arg(data.currentMapName);
		info += QString("置信度: %1\n").arg(data.confidence * 100, 0, 'f', 2);


		qDebug() << QString("本体温度: %1°C").arg(data.bodyTemperature, 0, 'f', 2);
		qDebug() << QString("位置的 X 坐标: %1m").arg(data.xCoordinate, 0, 'f', 3);
		qDebug() << QString("位置的 Y 坐标: %1m").arg(data.yCoordinate, 0, 'f', 3);
		qDebug() << QString("位置的朝向角度: %1").arg(data.orientationAngle, 0, 'f', 3);
		qDebug() << QString("电池电量: %1%").arg(data.batteryLevel * 100, 0, 'f', 2);
		qDebug() << QString("是否被阻挡: %1").arg(data.isBlocked ? "是" : "否");
		qDebug() << QString("是否在充电: %1").arg(data.isCharging ? "是" : "否");
		qDebug() << QString("运行模式: %1").arg(data.runMode == 0 ? "手动" : "自动");
		qDebug() << QString("地图载入状态: %1").arg(data.mapLoadStatus == 0 ? "成功" : (data.mapLoadStatus == 1 ? "失败" : (data.mapLoadStatus == 2 ? "未载入" : "载入中")));
		qDebug() << QString("当前的目标点 id: %1").arg(data.currentTargetId);
		qDebug() << QString("前进速度: %1m/s").arg(data.forwardSpeed, 0, 'f', 3);
		qDebug() << QString("转弯速度: %1rad/s").arg(data.turnSpeed, 0, 'f', 3);
		qDebug() << QString("电池电压: %1V").arg(data.batteryVoltage, 0, 'f', 2);
		qDebug() << QString("电流: %1A").arg(data.current, 0, 'f', 2);
		qDebug() << QString("当前任务状态: %1").arg(data.taskStatus == 0 ? "无任务" : (data.taskStatus == 1 ? "等待" : (data.taskStatus == 2 ? "前往导航点" : (data.taskStatus == 3 ? "暂停" : (data.taskStatus == 4 ? "完成" : (data.taskStatus == 5 ? "失败" : (data.taskStatus == 6 ? "退出" : "等待开/关门")))))));
		qDebug() << QString("当前定位方式: %1").arg(data.positioningMode == 0 ? "未知" : (data.positioningMode == 1 ? "二维码" : (data.positioningMode == 2 ? "磁定位" : (data.positioningMode == 3 ? "激光无反" : (data.positioningMode == 4 ? "激光有反" : (data.positioningMode == 5 ? "高反码" : "GPS"))))));
		qDebug() << QString("地图版本号: %1").arg(data.mapVersion);
		qDebug() << QString("保留字段: %1").arg(data.reserved3);
		qDebug() << QString("累计行驶里程: %1m").arg(data.totalMileage, 0, 'f', 2);
		qDebug() << QString("本次运行时间: %1ms").arg(data.currentRunTime, 0, 'f', 2);
		qDebug() << QString("累计运行时间: %1ms").arg(data.totalRunTime, 0, 'f', 2);
		qDebug() << QString("机器人定位状态: %1").arg(data.positioningStatus == 0 ? "失败" : (data.positioningStatus == 1 ? "成功" : (data.positioningStatus == 2 ? "定位中" : "定位完成")));
		qDebug() << QString("交通管制: %1").arg(data.trafficControl ? "是" : "否");
		qDebug() << QString("保留字段: %1").arg(data.reserved4);
		qDebug() << QString("地图数量: %1").arg(data.mapCount);
		qDebug() << QString("当前地图名称: %1").arg(data.currentMapName);
		qDebug() << QString("置信度: %1").arg(data.confidence * 100, 0, 'f', 2);
		return info;
	}

	if (data.commandCode == 0xAF) {
		info += QString("异常事件状态信息长度: %1\n").arg(data.abnormal_size);
		info += QString("动作状态长度: %1\n").arg(data.action_size);
		info += QString("信息数量: %1m\n").arg(data.Info_size);
		info += QString("机器人全局位置 x 坐标: %1\n").arg(data.GlobalX, 0, 'f', 3);
		info += QString("机器人全局位置 y 坐标: %1\n").arg(data.GlobalY, 0, 'f', 3);
		info += QString("机器人绝对车体方向角: %1\n").arg(data.GlobalA, 0, 'f', 3);
		info += QString("最后通过点 ID: %1\n").arg(data.endID);
		info += QString("最后通过段 ID: %1\n").arg(data.endParagraph);
		info += QString("点序列号: %1\n").arg(data.Dotserianumber);
		info += QString("置信度: %1\n").arg(data.confidenceAF);
		info += QString("定位状态: %1\n").arg(data.positioningStatus == 0 ? "失败" : (data.positioningStatus == 1 ? "成功" : (data.positioningStatus == 2 ? "定位中" : "定位成功")));
		info += QString("X 轴速度: %1\n").arg(data.LinearVelocity, 0, 'f', 3);
		info += QString("Y 轴速度: %1\n").arg(data.Acceleration, 0, 'f', 2);
		info += QString("角速度: %1A\n").arg(data.AngularVelocity, 0, 'f', 2);
		info += QString("工作模式: %1\n").arg(data.WorkMode == 0 ? "待机" : (data.WorkMode == 1 ? "手动" : (data.WorkMode == 2 ? "半自动" : (data.WorkMode == 3 ? "自动" : (data.WorkMode == 4 ? "示教" : (data.WorkMode == 5 ? "服务" : "维修"))))));
		info += QString("AGV 状态: %1\n").arg(data.AGVStatus == 0 ? "空闲" : (data.AGVStatus == 1 ? "运行" : (data.AGVStatus == 2 ? "暂停" : (data.AGVStatus == 3 ? "初始化中" : (data.AGVStatus == 4 ? "人工确认" : (data.AGVStatus == 5 ? "未初始化" : "导航失败"))))));
		info += QString("机器人能力集设置状态: %1\n").arg(data.AbilitySettings == 0 ? "未设置能力集" : "已设置能力集");
		info += QString("订单 ID: %1\n").arg(data.OrderID);
		info += QString("任务 KEY: %1\n").arg(data.OrderKEY);
		info += QString("点状态序列长度: %1\n").arg(data.PointStateSequence);
		info += QString("段状态序列长度: %1\n").arg(data.PathStateSequence);
		//中间待补充
		info += QString("电量百分比: %1\n").arg(data.Batterylevel, 0, 'f', 3);//FLOAT
		info += QString("电压: %1\n").arg(data.VoltageAF, 0, 'f', 3);
		info += QString("电流: %1\n").arg(data.ElectricCurrent, 0, 'f', 3);
		info += QString("充电情况: %1\n").arg(data.ChargingStatus == 0 ? "放电" : (data.ChargingStatus == 1 ? "充电" : (data.ChargingStatus == 2 ? "充满电" : "不是2")));
		//补充待写

		qDebug() << QString("异常事件状态信息长度: %1").arg(data.abnormal_size);
		qDebug() << QString("动作状态长度: %1").arg(data.action_size);
		qDebug() << QString("信息数量: %1").arg(data.Info_size);
		qDebug() << QString("机器人全局位置 x 坐标: %1").arg(data.GlobalX, 0, 'f', 3);
		qDebug() << QString("机器人全局位置 y 坐标: %1").arg(data.GlobalY, 0, 'f', 3);
		qDebug() << QString("机器人绝对车体方向角: %1").arg(data.GlobalA, 0, 'f', 3);
		qDebug() << QString("最后通过点 ID: %1").arg(data.endID);
		qDebug() << QString("最后通过段 ID: %1").arg(data.endParagraph);
		qDebug() << QString("点序列号: %1").arg(data.Dotserianumber);
		qDebug() << QString("置信度: %1").arg(data.confidenceAF);
		qDebug() << QString("定位状态: %1").arg(data.positioningStatus == 0 ? "失败" : (data.positioningStatus == 1 ? "成功" : (data.positioningStatus == 2 ? "定位中" : "定位成功")));
		qDebug() << QString("X 轴速度: %1").arg(data.LinearVelocity, 0, 'f', 3);
		qDebug() << QString("Y 轴速度: %1").arg(data.Acceleration, 0, 'f', 2);
		qDebug() << QString("角速度: %1").arg(data.AngularVelocity, 0, 'f', 2);
		qDebug() << QString("工作模式: %1").arg(data.WorkMode == 0 ? "待机" : (data.WorkMode == 1 ? "手动" : (data.WorkMode == 2 ? "半自动" : (data.WorkMode == 3 ? "自动" : (data.WorkMode == 4 ? "示教" : (data.WorkMode == 5 ? "服务" : "维修"))))));
		qDebug() << QString("AGV 状态: %1").arg(data.AGVStatus == 0 ? "空闲" : (data.AGVStatus == 1 ? "运行" : (data.AGVStatus == 2 ? "暂停" : (data.AGVStatus == 3 ? "初始化中" : (data.AGVStatus == 4 ? "人工确认" : (data.AGVStatus == 5 ? "未初始化" : "导航失败"))))));
		qDebug() << QString("机器人能力集设置状态: %1").arg(data.AbilitySettings == 0 ? "未设置能力集" : "已设置能力集");
		qDebug() << QString("订单 ID: %1").arg(data.OrderID);
		qDebug() << QString("任务 KEY: %1").arg(data.OrderKEY);
		qDebug() << QString("点状态序列长度: %1").arg(data.PointStateSequence);
		qDebug() << QString("段状态序列长度: %1").arg(data.PathStateSequence);
		//中间待补充
		qDebug() << QString("电量百分比: %1").arg(data.Batterylevel, 0, 'f', 3);//FLOAT
		qDebug() << QString("电压: %1").arg(data.VoltageAF, 0, 'f', 3);
		qDebug() << QString("电流: %1").arg(data.ElectricCurrent, 0, 'f', 3);
		qDebug() << QString("充电情况: %1").arg(data.ChargingStatus == 0 ? "放电" : (data.ChargingStatus == 1 ? "充电" : (data.ChargingStatus == 2 ? "充满电" : "不是2")));
		//补充待写

		return info;
	}

	return info;
}

//发送
void kcUDP::write_datakcUDP(QByteArray str)
{
	//if (MessageData != "" || STARTN)
	//{
	//	qDebug() << MessageData.toHex(' ').toUpper();
	//	udpSocket->writeDatagram(MessageData, QHostAddress(ip), port);
	//}
	qDebug() << "发送" << str.toHex(' ').toUpper();
	udpSocket->writeDatagram(str, QHostAddress(ip), port);
}

void kcUDP::AAdatakcUDP(bool DATA)
{
	m_taskReCall = DATA;
}
//密钥提取
QByteArray kcUDP::hexToQByteArray(QString& hexStr) {
	QByteArray byteArray;
	for (int i = 0; i < hexStr.length(); i += 3) {
		bool ok;
		quint8 byte = hexStr.mid(i, 2).toUInt(&ok, 16);
		if (ok) byteArray.append(byte);
	}
	return byteArray;
}
//变量ASCII码转换
QByteArray kcUDP::convertTo16Array(QString& input) {
	QByteArray byteArray = input.toUtf8();

	if (byteArray.size() < 16) {
		int originalSize = byteArray.size();
		byteArray.resize(16);
		// 手动填充0
		for (int i = originalSize; i < 16; ++i) {
			byteArray[i] = 0;
		}
	}
	return byteArray;
}
//转小端
QByteArray kcUDP::convertToLittleEndian(quint16 value) {
	QByteArray result;
	QDataStream stream(&result, QIODevice::WriteOnly);
	stream.setByteOrder(QDataStream::LittleEndian);
	stream << value;
	return result;
}
// 检测系统字节序
bool  kcUDP::isLittleEndian() {
	uint32_t value = 0x01020304;
	uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
	return bytes[0] == 0x04;
}

// double转换为小端字节序
QByteArray  kcUDP::toLittleEndian(double value) {
	QByteArray result(8, 0);
	if (isLittleEndian()) {
		memcpy(result.data(), &value, 8);
	}
	else {
		uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
		for (int i = 0; i < 8; ++i) {
			result[i] = bytes[7 - i];
		}
	}
	return result;
}

//切换定位为手动模式(0x11 手动) //已验证发送正常
QByteArray  kcUDP::SwitchTOmanual() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;
	quint8 AGVdata[4] = { 0,0,0,0 };
	MessageLenth = sizeof(AGVdata);
	QByteArray Lenth = convertToLittleEndian(MessageLenth);

	//qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_MODE_SWITCH) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2) + QByteArray(reinterpret_cast<const char*>(AGVdata), 4);;
	qDebug() << "切换手动" << data.toHex(' ');
	//MessageData = data;
	write_datakcUDP(data);
	return data;
}

//切换成自动模式 //已验证发送正常
QByteArray  kcUDP::SwitchTOautomatic() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;
	quint8 AGVdata[4] = { 1,0,0,0 };
	MessageLenth = sizeof(AGVdata);
	QByteArray Lenth = convertToLittleEndian(MessageLenth);

	//qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_MODE_SWITCH) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2) + QByteArray(reinterpret_cast<const char*>(AGVdata), 4);
	qDebug() << "切换自动" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//执行机器人手动定位 0x14//已验证发送弧度需要验证
QByteArray  kcUDP::ExecuteManually() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 24;//默认24
	QByteArray Lenth = convertToLittleEndian(MessageLenth);
	Tx = 0.05; Ty = 0.038; Ta = 0;

	QByteArray littleEndianX = toLittleEndian(Tx);
	QByteArray littleEndianY = toLittleEndian(Ty);
	QByteArray littleEndianA = toLittleEndian(Ta);

	//qDebug() << littleEndianX.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_ROBOT_MANUAL_POSITION) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2) + littleEndianX + littleEndianY + littleEndianA;
	qDebug() << "手动定位" << data.toHex(' ');
	write_datakcUDP(data);

	return data;
}

//查询机器人运行状态（0xAF）//已验证发送正常
QByteArray  kcUDP::CheckStatus() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 0;
	QByteArray Lenth = convertToLittleEndian(MessageLenth);

	//qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_CONFIRM_ROBOT_STATUS) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2);
	qDebug() << "状态检查" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//确认机器人位置（0x1F）//已验证发送正常
QByteArray  kcUDP::Confirmlocation() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 0;
	QByteArray Lenth = convertToLittleEndian(MessageLenth);
	qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_CONFIRM_ROBOT_POSITION) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2);
	qDebug() << "确认位置" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//获取机器人当前位置（命令码：0x15）//已验证发送正常
QByteArray kcUDP::GetLocation() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 0;
	QByteArray Lenth = convertToLittleEndian(MessageLenth);
	qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_GET_ROBOT_POSITION) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2);
	qDebug() << "获取位置" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//查询机器人运行状态（命令码：0x17）//已验证发送正常
QByteArray kcUDP::CheckRunningStatus() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 0;
	QByteArray Lenth = convertToLittleEndian(MessageLenth);
	qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_QUERY_ROBOT_STATUS) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2);
	qDebug() << "获取机器人运动状态" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//查询机器人导航状态（命令码：0x1D）//已验证发送正常
QByteArray kcUDP::NavigationStatus() {
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	MessageLenth = 0;
	QByteArray Lenth = convertToLittleEndian(MessageLenth);
	qDebug() << sumA.toHex(' ');
	data = devicekey + QByteArray(1, Version) + QByteArray(1, REQUEST) + sumA + QByteArray(1, ServiceM)
		+ QByteArray(1, ORDER_QUERY_NAVIGATION_STATUS) + QByteArray(1, ExecuteREQUEST) + QByteArray(1, Reserved1) + Lenth
		+ QByteArray(reinterpret_cast<const char*>(Reserved2), 2);
	qDebug() << "查询机器人导航状态" << data.toHex(' ');
	write_datakcUDP(data);
	return data;
}

//导航控制（命令码：0x16）
QByteArray kcUDP::NavigationControl()
{
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	// 导航控制协议数据
	/*0：开始导航1：取消导航2：暂停导航3：继续导航4：创建导航任务并暂停导航
	5：返航(目前无效)6：从路径外导航到路径上7：立即取消*/
	quint8 data1 = 0;  // 操作类型
	/*0：导航到路径点1：导航到路径上的点2：自由导航到地图上的点*/
	quint8 data2 = 0;  // 导航方式
	/*0：不指定导航路径1：指定导航路径*/
	quint8 data3 = 0;  // 是否指定导航路径
	/*0：不启用1：启用交通管理，所有点和路径需要申请后使用*/
	quint8 data4 = 0;  // 是否启用交通管理
	/*导航方式为 0 时使用*/
	quint8 data5[8] = { 0 };  // 路径点ID 
	/*导航方式为 1 时使用*/
	quint16 data6 = 0;  // 目标路径起点ID
	quint16 data7 = 0;  // 目标路径终点ID
	/*导航方式为 1 或 2 时使用*/
	float data8 = 0.0f;  // 目标点坐标x
	float data9 = 0.0f;  // 目标点坐标y
	float data10 = 0.0f; // 目标点坐标theta

	quint8 data11[2] = { 0 };  // 保留字段
	/*导航指定路径时使用*/
	quint16 data12 = 0;  // 指定路径的路径点数量
	quint16 data13[128] = { 0 };  // 指定路径的所有路径点ID

	quint8 data14[12] = { 0 };  // 保留字段

	quint8 data15 = 0;//禁止通行路径的路径数量，不超过 32
	quint8 data16[3] = { 0 };  // 保留字段
	/*禁止通行路径[0]：第 0 条路径起点 ID[1]：第 0 条路径终点 ID[2]：第 1 条路径起点 ID*/
	quint16 data17[64] = { 0 };

	// 示例：开始导航，导航到路径上的点
	data1 = 0;  // 开始导航
	data2 = 0;  // 导航到路径点
	data3 = 1;  // 指定导航路径
	data4 = 0;  // 不启用交通管理
	strcpy((char*)data5, "5");//路径点ID
	data6 = 0;  // 路径起点ID
	data7 = 0;  // 路径终点ID
	data8 = 0;  // 目标点x坐标
	data9 = 0;  // 目标点y坐标
	data10 = 0;  // 目标点角度

	data12 = 5;  // 4个路径点

	// 设置路径点ID数组
	data13[0] = 1;  // 起点
	data13[1] = 2;  // 中间点1
	data13[2] = 3;  // 中间点2
	data13[3] = 4;  // 终点
	data13[4] = 5;

	// 构建SenData
	QByteArray SenData;
	QDataStream stream(&SenData, QIODevice::WriteOnly);
	stream.setByteOrder(QDataStream::LittleEndian);

	// 0x00: 操作类型
	stream << data1;
	// 0x01: 导航方式
	stream << data2;
	// 0x02: 是否指定导航路径
	stream << data3;
	// 0x03: 是否启用交通管理
	stream << data4;
	// 0x04-0x0B: 路径点ID (8字节)
	stream.writeRawData(reinterpret_cast<const char*>(data5), 8);
	// 检查当前位置，应该是0x0C
	if (SenData.size() != 0x0C) {
		qDebug() << "警告: 路径点ID后位置应为0x0C，实际为" << SenData.size();
	}
	// 0x0C: 目标路径起点ID
	stream << data6;
	// 0x0E: 目标路径终点ID
	stream << data7;
	// 检查当前位置，应该是0x10
	if (SenData.size() != 0x10) {
		qDebug() << "警告: 路径终点ID后位置应为0x10，实际为" << SenData.size();
	}
	stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
	// 0x10: 目标点坐标x
	stream << data8;
	// 0x14: 目标点坐标y
	stream << data9;
	// 0x18: 目标点坐标theta
	stream << data10;
	// 检查当前位置，应该是0x10
	if (SenData.size() != 0x1C) {
		qDebug() << "警告: 路径终点ID后位置应为0x1C，实际为" << SenData.size();
	}
	// 0x1C: 保留字段 (2字节)
	stream.writeRawData(reinterpret_cast<const char*>(data11), 2);
	// 0x1E: 指定路径的路径点数量
	stream << data12;
	// 0x20-0x11F: 指定路径的所有路径点ID (128个，每个2字节)
	for (int i = 0; i < 128; ++i) {
		stream << data13[i];
	}
	// 0x120-0x12B: 保留字段 (12字节)
	stream.writeRawData(reinterpret_cast<const char*>(data14), 12);
	// 0x12C: 禁止通行路径的路径数量，不超过 32
	stream << data15;
	// 0x12D-0x12B: 保留字段 (3字节)
	stream.writeRawData(reinterpret_cast<const char*>(data16), 3);
	// 0x130 禁止通行路径
	//stream.writeRawData(reinterpret_cast<const char*>(data17), 64);
	for (int i = 0; i < 64; ++i) {
		stream << data17[i];
	}

	// 验证SenData大小
	qDebug() << "SenData大小:" << SenData.size() << "字节";

	// 构建完整数据包
	MessageLenth = SenData.size();
	QByteArray Lenth = convertToLittleEndian(MessageLenth);

	qDebug() << "sumA:" << sumA.toHex(' ');

	data = devicekey +
		QByteArray(1, Version) +
		QByteArray(1, REQUEST) +
		sumA +
		QByteArray(1, ServiceM) +
		QByteArray(1, ORDER_NAVIGATION_CONTROL) +
		QByteArray(1, ExecuteREQUEST) +
		QByteArray(1, Reserved1) +
		Lenth +
		QByteArray(reinterpret_cast<const char*>(Reserved2), 2) +
		SenData;

	qDebug() << "导航发送数据:" << data.toHex(' ');
	write_datakcUDP(data);

	return data;
}

QByteArray kcUDP::NavigationControlBack()
{
	QByteArray data;
	QByteArray sumA = convertToLittleEndian(SequenceA);
	SequenceA++;

	// 导航控制协议数据
	/*0：开始导航1：取消导航2：暂停导航3：继续导航4：创建导航任务并暂停导航
	5：返航(目前无效)6：从路径外导航到路径上7：立即取消*/
	quint8 data1 = 0;  // 操作类型
	/*0：导航到路径点1：导航到路径上的点2：自由导航到地图上的点*/
	quint8 data2 = 0;  // 导航方式
	/*0：不指定导航路径1：指定导航路径*/
	quint8 data3 = 0;  // 是否指定导航路径
	/*0：不启用1：启用交通管理，所有点和路径需要申请后使用*/
	quint8 data4 = 0;  // 是否启用交通管理
	/*导航方式为 0 时使用*/
	quint8 data5[8] = { 0 };  // 路径点ID 
	/*导航方式为 1 时使用*/
	quint16 data6 = 0;  // 目标路径起点ID
	quint16 data7 = 0;  // 目标路径终点ID
	/*导航方式为 1 或 2 时使用*/
	float data8 = 0.0f;  // 目标点坐标x
	float data9 = 0.0f;  // 目标点坐标y
	float data10 = 0.0f; // 目标点坐标theta

	quint8 data11[2] = { 0 };  // 保留字段
	/*导航指定路径时使用*/
	quint16 data12 = 0;  // 指定路径的路径点数量
	quint16 data13[128] = { 0 };  // 指定路径的所有路径点ID

	quint8 data14[12] = { 0 };  // 保留字段

	quint8 data15 = 0;//禁止通行路径的路径数量，不超过 32
	quint8 data16[3] = { 0 };  // 保留字段
	/*禁止通行路径[0]：第 0 条路径起点 ID[1]：第 0 条路径终点 ID[2]：第 1 条路径起点 ID*/
	quint16 data17[64] = { 0 };

	// 示例：开始导航，导航到路径上的点
	data1 = 0;  // 开始导航
	data2 = 0;  // 导航到路径点
	data3 = 1;  // 指定导航路径
	data4 = 0;  // 不启用交通管理
	strcpy((char*)data5, "1");//路径点ID
	data6 = 0;  // 路径起点ID
	data7 = 0;  // 路径终点ID
	data8 = 0;  // 目标点x坐标
	data9 = 0;  // 目标点y坐标
	data10 = 0;  // 目标点角度

	data12 = 5;  // 4个路径点

	// 设置路径点ID数组
	data13[0] = 5;  // 起点
	data13[1] = 4;  // 中间点1
	data13[2] = 3;  // 中间点2
	data13[3] = 2;  // 终点
	data13[4] = 1;

	// 构建SenData
	QByteArray SenData;
	QDataStream stream(&SenData, QIODevice::WriteOnly);
	stream.setByteOrder(QDataStream::LittleEndian);

	// 0x00: 操作类型
	stream << data1;
	// 0x01: 导航方式
	stream << data2;
	// 0x02: 是否指定导航路径
	stream << data3;
	// 0x03: 是否启用交通管理
	stream << data4;
	// 0x04-0x0B: 路径点ID (8字节)
	stream.writeRawData(reinterpret_cast<const char*>(data5), 8);
	// 检查当前位置，应该是0x0C
	if (SenData.size() != 0x0C) {
		qDebug() << "警告: 路径点ID后位置应为0x0C，实际为" << SenData.size();
	}
	// 0x0C: 目标路径起点ID
	stream << data6;
	// 0x0E: 目标路径终点ID
	stream << data7;
	// 检查当前位置，应该是0x10
	if (SenData.size() != 0x10) {
		qDebug() << "警告: 路径终点ID后位置应为0x10，实际为" << SenData.size();
	}
	stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
	// 0x10: 目标点坐标x
	stream << data8;
	// 0x14: 目标点坐标y
	stream << data9;
	// 0x18: 目标点坐标theta
	stream << data10;
	// 检查当前位置，应该是0x10
	if (SenData.size() != 0x1C) {
		qDebug() << "警告: 路径终点ID后位置应为0x1C，实际为" << SenData.size();
	}
	// 0x1C: 保留字段 (2字节)
	stream.writeRawData(reinterpret_cast<const char*>(data11), 2);
	// 0x1E: 指定路径的路径点数量
	stream << data12;
	// 0x20-0x11F: 指定路径的所有路径点ID (128个，每个2字节)
	for (int i = 0; i < 128; ++i) {
		stream << data13[i];
	}
	// 0x120-0x12B: 保留字段 (12字节)
	stream.writeRawData(reinterpret_cast<const char*>(data14), 12);
	// 0x12C: 禁止通行路径的路径数量，不超过 32
	stream << data15;
	// 0x12D-0x12B: 保留字段 (3字节)
	stream.writeRawData(reinterpret_cast<const char*>(data16), 3);
	// 0x130 禁止通行路径
	//stream.writeRawData(reinterpret_cast<const char*>(data17), 64);
	for (int i = 0; i < 64; ++i) {
		stream << data17[i];
	}

	// 验证SenData大小
	qDebug() << "SenData大小:" << SenData.size() << "字节";

	// 构建完整数据包
	MessageLenth = SenData.size();
	QByteArray Lenth = convertToLittleEndian(MessageLenth);

	qDebug() << "sumA:" << sumA.toHex(' ');

	data = devicekey +
		QByteArray(1, Version) +
		QByteArray(1, REQUEST) +
		sumA +
		QByteArray(1, ServiceM) +
		QByteArray(1, ORDER_NAVIGATION_CONTROL) +
		QByteArray(1, ExecuteREQUEST) +
		QByteArray(1, Reserved1) +
		Lenth +
		QByteArray(reinterpret_cast<const char*>(Reserved2), 2) +
		SenData;

	qDebug() << "导航发送数据:" << data.toHex(' ');
	write_datakcUDP(data);

	return data;
}